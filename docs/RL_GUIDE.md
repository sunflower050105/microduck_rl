# Learning RL through the Microduck repo

A study guide to how this repo trains a small biped robot with reinforcement
learning: the environment, the MDP pieces (observations, actions, rewards,
terminations, curricula), the algorithm, and what is specific to each task.

Everything below is taken from the code in `src/mjlab_microduck/`. File and
function names are given so you can jump into the source. Where a value comes
from mjlab's base template (not vendored in this repo), it is marked
**(mjlab template)**.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Environment setup](#2-environment-setup)
3. [Anatomy of one environment (the MDP)](#3-anatomy-of-one-environment-the-mdp)
4. [The learning algorithm (same for every task)](#4-the-learning-algorithm-same-for-every-task)
5. [Reward engineering: the recipe and its rules](#5-reward-engineering-the-recipe-and-its-rules)
6. [Curriculum learning](#6-curriculum-learning)
7. [Sim2real machinery](#7-sim2real-machinery)
8. [Task-by-task walkthrough](#8-task-by-task-walkthrough)
9. [Task comparison table](#9-task-comparison-table)
10. [How to build and debug a new task](#10-how-to-build-and-debug-a-new-task)
11. [Suggested study path](#11-suggested-study-path)

---

## 1. The big picture

```
 Onshape CAD ──► MJCF robot models ──► mjlab env (MuJoCo Warp, thousands of parallel envs on GPU)
                                            │
                       PPO (rsl_rl) learns a small MLP policy: obs(61) → action(14)
                                            │
                          export.py → ONNX (obs normalizer baked in)
                                            │
                     publish → Hugging Face Hub → runtime on the real robot
```

- **Robot:** Microduck, ~800 g, ~25 cm, 14 Dynamixel XL330 servos (5 per leg,
  4 for neck/head).
- **Simulator / framework:** [mjlab](https://github.com/mujocolab/mjlab) on
  MuJoCo Warp. It gives you a *manager-based* RL env: you declare observation,
  reward, event, termination, command and curriculum "terms" and the framework
  wires them into a vectorised env.
- **Algorithm:** PPO from `rsl_rl`. It is the same algorithm and almost the same
  hyper-parameters for every task. **The tasks differ in the MDP, not in the
  learner.** This is the most important thing to understand about this repo.
- **Control rate:** policies run at 50 Hz (`AGENTS.md`).
- **Goal is sim2real**, so much of the code exists to make the simulator
  imperfect in realistic ways (actuator model, delays, noise, backlash, DR).

---

## 2. Environment setup

### 2.1 Requirements

- A CUDA GPU (training runs through MuJoCo Warp) and [`uv`](https://docs.astral.sh/uv/).
- Python `>=3.12,<3.13` (pinned in `pyproject.toml`, because `bam` requires it).
- Key pinned dependencies: `mjlab==1.3.0`, `warp-lang==1.12.0`,
  `better-actuator-models` (imported as `bam`), `torch==2.9.1`, `onnxruntime`.
- On ARM (DGX Spark / Jetson), torch is routed to the CUDA index by
  `[tool.uv.sources]`; see the "Wheels are per-architecture" note in `AGENTS.md`.
  First sync is ~2 GB, so `export UV_HTTP_TIMEOUT=600`.

### 2.2 Commands

```bash
uv sync                                   # install (a fresh sync is the ground truth; HF Jobs run one)
uv run list-envs                          # live task registry

# ALWAYS smoke-test first: 64 envs, 5 iterations. Catches ~95% of config errors.
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# real training
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# no local GPU: run on Hugging Face Jobs
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs

# resume
uv run train <TASK> --agent.load-checkpoint model_XXXX.pt --agent.resume True

# watch, export, rehearse, publish
uv run play <TASK> --wandb-run-path <entity/project/run_id>
uv run scripts/export.py <TASK> --wandb-run-path <...>          # → ONNX
uv run scripts/infer_policy.py --walking out.onnx               # CPU MuJoCo rehearsal
uv run publish --task <TASK> --wandb-run-path <...> --checkpoint N --repo <user>/microduck-<name> --kind episodic --duration-s 4.0

uv run --with pytest pytest tests/                              # CPU-only cfg / mdp tests
```

Logs go to `logs/<experiment_name>/`; metrics to the wandb project `mjlab_microduck`.

### 2.3 How tasks get registered

`src/mjlab_microduck/tasks/__init__.py` calls mjlab's `register_mjlab_task(...)`
once per task with four things:

| Argument | Meaning |
|---|---|
| `env_cfg` | the training env config, built by a `make_microduck_*_env_cfg()` factory |
| `play_env_cfg` | the same env with `play=True` (few envs, shorter push interval, viewer-friendly) |
| `rl_cfg` | the PPO / network config (`Microduck*RlCfg`) |
| `runner_cls` | `MicroduckOnPolicyRunner` (a thin subclass of mjlab's velocity runner) |

`pyproject.toml` exposes this to mjlab through the `mjlab.tasks` entry point,
so `train`/`play` can find the tasks by id.

Each base task also has a **`-Backlash-` twin** generated by
`tasks/backlash.py::make_backlash_variant` (see §7.3).

---

## 3. Anatomy of one environment (the MDP)

An RL problem is an MDP: states, actions, transitions, rewards, discount. Here is
where each piece lives.

### 3.1 Robot and physics (the "transition function")

- **Models** (`robot/microduck/`): MJCF files exported from Onshape. Families:
  - `walk` — only the feet collide with the world (fast, used for walking).
  - `groundcontact` — a curated set of parts can touch the floor (soles, legs,
    trunk shells, head, jaw...). Used when the body may lie on the ground.
  - `allcollisions` — every part collides; XL330 housings are named
    `*_servo_collision` so a sensor can detect "a servo hit the floor".
  - `*_rollers` — passive wheels under the feet (roller skates).
  - `*_backlash` — each servo gets an extra unactuated hinge with ±1° of play.
- **Home pose** (`HOME_FRAME` in `robot/microduck_constants.py`): the nominal
  standing joint angles. Almost every "pose" reward is measured relative to it.
- **Joint layout** (14 servos): `0–4` left leg (hip_yaw, hip_roll, hip_pitch,
  knee, ankle), `5–8` neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  `9–13` right leg. Passive joints are named `passive_*`; on roller/backlash
  models they interleave with the servos, so code must resolve joints **by name**
  (`_servo_joint_ids`, `_servo_joint_pos` helpers in `mdp.py`), never by index.
- **Actuators are the BAM model** (`actuator/friction_dr_bam.py`): a
  voltage-controlled XL330 model with load-dependent friction, battery-voltage
  sag and a 3–6 step command delay (`_BAM_ACTUATOR_KWARGS`: `kp_fw=200`,
  `vin_range=(6.5, 8.2)`, `delay 3–6`). It is far closer to the real servo than
  a plain MuJoCo position actuator.
  `FrictionDRBamActuator` adds a per-env `friction_scale` so friction can be
  domain-randomised; `BacklashEncoderBamActuator` makes the servo PD loop read
  its encoder *through* the backlash, like the real hardware.

### 3.2 Action space

14-D: a **joint-position target** for every servo (`JointPositionActionCfg`,
`scale = 1.0`, offset from the home pose in the mjlab template). The BAM actuator
turns that target into torque. Policies are **unfiltered**: no smoothing between
network output and servo. Smoothness must therefore be learned through rewards
(`action_rate_l2`, `joint_torque_rate_l2`).

### 3.3 Observation space (the contract)

The actor sees **61 numbers**, identical across every task so policies can be
hot-swapped at runtime:

| Slice | Content | Size |
|---|---|---|
| `[0:3]` | `base_ang_vel` (IMU gyro, body frame) | 3 |
| `[3:6]` | `projected_gravity` (tilt) | 3 |
| `[6:20]` | `joint_pos` relative to home (+ encoder bias) | 14 |
| `[20:34]` | `joint_vel` | 14 |
| `[34:48]` | `last_action` | 14 |
| `[48:51]` | **twist command** `[vx, vy, ωz]` (or a task-specific meaning, see below) | 3 |
| `[51:55]` | **head_pose command** (neck_pitch, head_pitch, head_yaw, head_roll deltas) | 4 |
| `[55:61]` | **body_pose command** (x, y, z, roll, pitch, yaw deltas) | 6 |

Rules that follow from this:

- A task that does not use a command slot **zero-pads it** (keeps the term,
  samples a tiny range). It never deletes it.
- The **twist slot is reused** as a generic 3-number command channel:
  - Velocity tasks: `[vx, vy, ωz]`.
  - Sit/Stand: `[sit_flag, 0, 0]`. All zeros = "stand".
  - Ground-pick, Roller-crouch, Spin: `[cos 2πφ, sin 2πφ, 0]` (a phase clock).
  - Roller velocity: `cmd_x` means coast / push / brake.
- Every command slot keeps a **small non-zero range from step 0** so its input
  weights never go dead.

**Asymmetric actor–critic.** The critic sees extra *privileged* information the
real robot does not have: true base linear velocity, foot contact forces, foot
heights, foot air-time (for BallKick also ball position/velocity). The critic is
only used in training, so this is free and makes the value function much better.
Critic-only terms use `_safe` (NaN-sanitised) variants because a single NaN in a
critic obs kills the whole run.

**Sensor realism applied to the actor only:**

| Effect | Value (velocity task) |
|---|---|
| IMU noise | ang_vel ±0.03, gravity ±0.01 |
| Joint noise | pos ±0.001 rad, vel ±0.25 |
| IMU delay | 0–1 control steps (≈ ±20 ms) |
| joint_vel lag | exactly 1 step (Dynamixel moving-average velocity) |
| IMU mounting error | per-env random rotation up to 6°, zero-centred |
| Encoder bias | per-env constant offset ±0.015 rad (±0.86°) |

### 3.4 Rewards

Each reward is a `RewardTermCfg(func=..., weight=..., params=...)`. The total
per-step reward is `Σ weight · func(env)`. All custom functions live in
`tasks/mdp.py` (7,400 lines, grouped by task). See §5.

### 3.5 Terminations and episodes

- **Time-out** (`cfg.episode_length_s`): ~20 s for locomotion (mentioned in the VelStand docstring; exact template default not read); episodic
  tricks are short (StandUp 6 s, Roulade 5 s, BallKick 5 s, SitStand 12 s).
- **`nan_state`** (`robot_state_is_nan`): terminates any env whose physics went
  NaN so a bad state never reaches the network.
- Task-specific: `fell_over` (tilt limit), `fallen_too_long` (VelStand), `fell_into_void`
  (slope).

### 3.6 Events (randomisation and resets)

Events run at `startup`, `reset` or `interval`. They are how domain randomisation
(DR), pushes and special spawn states are implemented. Details in §7.

### 3.7 Commands

Commands are sampled by the command manager and resampled periodically:

- `VelocityCommandCommandOnlyCfg` — twist command with `rel_standing_envs`
  (fraction of exact-zero commands) and `rel_turn_in_place_envs` buckets.
- `UniformPoseCommandCfg` — head_pose (4-D) and body_pose (6-D) deltas, resampled every 2–5 s.
- `GroundPickPhaseCommandCfg` — a phase clock `φ ∈ [0,1)` encoded as `[cos, sin, 0]`.
- `SitStandCommandCfg` — a sit/stand flag that flips every few seconds.

---

## 4. The learning algorithm (same for every task)

### 4.1 PPO with an actor–critic MLP

Every task uses `RslRlOnPolicyRunnerCfg` with these values (from
`MicroduckRlCfg` and its siblings):

| Setting | Value | What it does |
|---|---|---|
| Actor / critic | MLP `512 → 256 → 128`, ELU | policy and value function (separate networks) |
| `obs_normalization` | `True` | running mean/std normalisation of inputs |
| Action distribution | Gaussian, `init_std=1.0`, `std_type="scalar"` | one learnable std per action dim, state-independent |
| `num_steps_per_env` | 24 | rollout length per env per iteration |
| Envs | 4096 (typical) | ⇒ 98,304 transitions per iteration |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 | passes over each batch |
| `clip_param` | 0.2 | PPO ratio clip |
| `use_clipped_value_loss` | True, `value_loss_coef=1.0` | |
| `gamma`, `lam` | 0.99, 0.95 | discount and GAE λ |
| `entropy_coef` | 0.01 | exploration bonus (0.03 for rollers) |
| `learning_rate` | 1e-3, `schedule="adaptive"`, `desired_kl=0.01` | LR is auto-adjusted to keep the policy-update KL near 0.01 |
| `max_grad_norm` | 1.0 | |
| `save_interval` | 250 iterations | |

Study notes:

- **On-policy**: each iteration collects fresh data with the current policy, does
  a few epochs of updates, throws the data away. This is why massive parallel
  simulation matters.
- **GAE(λ=0.95)** trades bias vs variance in the advantage estimate.
- **Adaptive LR**: if measured KL > 2·desired_kl the LR shrinks; if < desired_kl/2
  it grows. That is why `learning_rate=1e-3` is only a starting point.
- **Why obs normalisation matters for deployment**: the normaliser is a
  *stateful part of the policy*. `scripts/export.py` bakes it into the ONNX. In
  the viewer the normaliser is applied anyway, so a wrong export looks fine in
  sim and fails on the robot.
- **Iteration ↔ step conversion**: curricula are indexed in *env steps* =
  `iteration × 24` (`NUM_STEPS_PER_ENV`).

### 4.2 Variants of the algorithm used in the repo

| Variant | Where | What is different |
|---|---|---|
| Plain PPO (`PpoWithSymmetryCfg`, symmetry **off**) | almost every task | the config class carries an optional mirror-loss; it is disabled everywhere (`ENABLE_SYMMETRY=False`) |
| **Symmetry mirror loss** (available, off) | `tasks/symmetry.py` | adds a loss that makes the policy equivariant under left–right reflection (permutation + sign tables for the 61-D obs). Never for asymmetric tasks (e.g. BallKick) |
| **PPO + fallen-gated expert behaviour cloning** (`PpoWithExpertBc`) | **VelStand only** (`tasks/distill.py`) | after each PPO update, a BC pass regresses the student's actions onto a frozen *stand-up expert* on **fallen** frames and onto the frozen *walk* checkpoint on **upright** frames |
| **Warm start** | VelStand, some others | load weights/normaliser/optimizer from another run, with `MICRODUCK_WARM_START=1` so the step counter and iteration restart at 0 |

The VelStand BC story is a great case study: two RL-only warm-started runs never
learned to get up because the policy's action noise was too small for random
flailing to produce even a partial rise, so the potential-based recovery rewards
never paid anything (no gradient). Distilling an existing expert fixed
discovery; RL then shaped the fall and the last mile.

### 4.3 Reading a training run

From `AGENTS.md`, watch every iteration for:

1. Mean reward rising **and** episode length behaving as the task demands.
2. **Every penalty term ≤ 0** in `Episode_Reward/<term>` (the sign check, §5.2).
3. The **main task term** actually growing (total reward can rise on
   regularisers alone while the trick never happens).
4. Remember `Episode_Reward/<term>` logs the *weighted* value: a term at weight 0
   reads 0 regardless of behaviour.
5. When a metric steps **down exactly at a curriculum stage boundary**, the pacing
   is wrong: stretch the stage or introduce the thing later.

Budgets: simple tricks ≈ 1000 iters; gaits and recovery 4000–6000.

---

## 5. Reward engineering: the recipe and its rules

### 5.1 The velocity (walking) reward stack

Built on mjlab's velocity template **(mjlab template)**, then re-tuned in
`microduck_velocity_env_cfg.py`:

| Term | Weight | Intent |
|---|---|---|
| `track_linear_velocity` | +2.0 (std √0.1) | follow commanded vx, vy |
| `track_angular_velocity` | +2.0 (std √0.5) | follow commanded yaw rate |
| `head_pose_tracking` | +2.0 (std 0.5) | follow commanded head pose (mean over 4 joints of a Gaussian) |
| `pose` | +1.0 | stay near home pose; **tight when standing, loose when walking** (`std_standing` vs `std_walking`); legs only |
| `upright` | +2.0 (std² 0.05) | keep the trunk level |
| `air_time` | +3.0, window 0.125–0.30 s | reward proper stepping |
| `foot_clearance`, `foot_swing_height` | template weights, target 2 cm | lift the feet |
| `foot_slip` | −0.1 | deliberately weak: strong values blocked pivot turning |
| `action_rate_l2` | −0.1 → −1.0 (curriculum) | smoothness |
| `body_ang_vel`, `angular_momentum` | −0.05, −0.02 | damp trunk spin |
| `self_collisions` | −1.0 | keep legs out of the trunk |
| `head_pose_bias` | 0 → 3.0 (curriculum, self-negating so positive weight) | penalise DC head droop only (see below) |

### 5.2 The rules (each learned the hard way; they are in `AGENTS.md`)

**Sign convention.** There are two penalty styles. mjlab-style cost functions
return `≥ 0`, so they take a **negative** weight. Microduck *self-negating*
functions (`*_penalty`, `*_l1`, returning `≤ 0`) take a **positive** weight.
A negative weight on a self-negating function double-negates into a *reward for
the violation* and the policy will farm it. Look at the table in §8: a positive
weight next to a `*_penalty` function (e.g. `gentle_rise = +0.005` with
`trunk_vertical_accel_penalty`) is **correct**, not a bug. Check on every run
that every penalty logs ≤ 0.

**RL optimises the letter of the reward.** Any under-specified degree of freedom
will be exploited. Encode "what counts as the maneuver" in **hard state gates**
(contact, orientation, latches), not in small penalty nudges.

**No jackpots.** Any "reach X" reward must be rate-limited or slewed. Otherwise
arriving early at a goal that pays per step buys arbitrary violence.

**Never gate a positive reward on being in a bad state** (fallen, low): the
policy parks in the cheapest qualifying pose and farms it. Use **potential-based
shaping** instead (pay Δprogress; holding pays zero):

```python
# tasks/mdp.py
def upright_progress(env, ...):   # Δ cos(tilt) per step
def height_progress(env, ...):    # Δ min(trunk_z, ceiling) per step
```

Rising pays, falling costs, standing still pays exactly 0, so nothing can be
farmed. Potential-based shaping is policy-invariant (Ng et al. 1999), so it speeds
up learning without changing the optimum.

**Multiplicative composites beat additive sums at goal states.** Additive
stacks have compromise basins (80% of every term via a lean). A product of
Gaussians collapses if any factor is bad:

```python
# tasks/mdp.py :: standing_composite_score
score = exp(-((z - z_target)/σ_h)²) * exp(-tilt²/σ_u²) * exp(-pose_err²/σ_p²)
```

**Tracking-Gaussian std ≈ the error you still care about.** But before tightening
ask whether the error is *escapable*. Example from this repo: a tight head-tracking
std made the policy stand still because a 280 g head *must* oscillate while
walking. The fix (`head_pose_bias_penalty`) is an L1 on a 1-second EMA of the
error, which charges the escapable DC droop and lets oscillation cancel.

**Two kinds of regularisers.**
- *Motion-blockers* (`body_ang_vel`, `angular_momentum`, pose std) penalise things
  dynamic motions physically need. Keep them low for dynamic tasks.
- *Smoothness* (`action_rate`, `torque_rate`) damps jitter without blocking slow
  large motions. Weight them, but **introduce them after skill discovery**, because
  an "attempt tax" active during exploration makes "do nothing" the winning policy.

**Compare reward mass, not weights, when copying regularisers** between tasks.
PPO sees relative advantage; the same `action_rate` weight is 4× weaker against a
4×-larger positive stack.

**Reward the same quantity the actor observes.** If an observation is remapped
to a sensor view (backlash encoder, encoder bias), any tracking reward on it
must use the same view, otherwise the policy is punished for correcting what it
sees. (`head_pose_tracking` does this for backlash.)

**Joint limit parking:** use a qpos-side limit-proximity penalty; command-side
penalties do not work because the actuator ctrl range is intentionally wide.

**Command inputs need to be alive.** A command that is never non-zero has dead
weights forever. Zero-command behaviour must be trained explicitly with
`rel_standing_envs`; rare regions (turn in place) need their own bucket
(`rel_turn_in_place_envs = 0.15`).

---

## 6. Curriculum learning

Three flavours, all implemented as mjlab `CurriculumTermCfg`:

1. **Reward-weight schedules** — `microduck_mdp.reward_weight` with
   `weight_stages=[{"step": ..., "weight": ...}]`. It is a *step function*, not
   an interpolation, so ramps are discretised. Example (velocity):

   ```
   action_rate_l2:   -0.1 (0) → -0.2 (iter 500) → -0.4 (750) → -0.6 (1000) → -0.8 (1250) → -1.0 (1500)
   head_pose_bias:    0   (0) →  1.0 (600) → 2.0 (1000) → 3.0 (1500)
   ```
2. **Parameter-range curricula** — command ranges (`pose_command_range_curriculum`),
   CoM DR ranges (`com_range_curriculum`), standing-env fraction
   (`standing_envs_curriculum`), push magnitude, wheel friction.
3. **Spawn-state curricula** — start episodes in harder or partial states
   (`ground_state_mix`, `prone_init_prob`, `roulade_spawn_mix`,
   `terrain_levels_slope`). *Reverse curriculum* (spawn partway through the
   maneuver) is the standard fix for "learns the start, never the last mile".

Principles: phase-align stages with what the policy has actually learned; never
harden before the current slice consolidates; never introduce taxes before the
skill exists. Always mutate term configs through managers
(`env.reward_manager.get_term_cfg(name)`), because `env.cfg` is a deep copy at
init and writes to it are silent no-ops.

---

## 7. Sim2real machinery

### 7.1 Domain randomisation (velocity-parity across tasks)

| DR | Range | Mechanism |
|---|---|---|
| Trunk CoM | ±3 mm → ±15 mm (curriculum) | `dr.body_ipos`, reset |
| Head CoM | ±3 mm → ±10 mm | same |
| Mass + inertia | ×0.95–1.05 | `dr.pseudo_inertia`, startup |
| Joint friction | ×0.9–1.1 | scales BAM's `friction_scale` (`dof_frictionloss` is zeroed under BAM, so randomising it would silently do nothing) |
| Armature | ×0.9–1.1 | `dr.joint_armature` |
| Foot friction | 0.7–1.3 | `foot_friction` event |
| Battery voltage / sag | 6.5–8.2 V, gain 0–0.2 | BAM actuator cfg |
| Pushes | ±0.3 m/s every 3–6 s | `push_by_setting_velocity` |
| IMU mounting | ≤6° | observation-level rotation |
| Encoder bias | ±0.015 rad | actor `joint_pos` only |
| KP/KD | off | disabled by choice |

**DR must not accumulate across resets.** In mjlab 1.3.0 `dr.*` ops with
`operation="add"/"scale"` re-read compile-time defaults, so they do not. Custom DR
must restore-then-apply. An accumulating CoM randomiser once degraded every long
run for months.

### 7.2 Terrain

Rough variants use a custom generator (`MICRODUCK_ROUGH_TERRAINS_CFG`): flat 25%,
stairs ≤1.5 cm 25%, random grid ≤1 cm 30%, gentle slopes 1.7°–5.7° 20%. Steps are
tiny because the robot can only lift its feet ~1–2 cm. Rough envs soften contact
solref and raise solver iterations to avoid NaN at box edges.

### 7.3 Backlash variants

`make_backlash_variant` swaps in the backlash robot, remaps `joint_pos/vel` obs to
the encoder view (`qpos[servo] + qpos[backlash]`), and scopes `dof_pos_limits` to
servo joints. Obs and action dimensions are unchanged, so export and runtime need
no change. A backlash task **must mirror its base task's robot model** so A/B
comparisons are not confounded.

### 7.4 Deployment path

`export.py` (ONNX, normaliser baked in) → `infer_policy.py` (CPU MuJoCo rehearsal
with BAM actuators) → `publish` (schema-2 manifest + smoke gate, uploads
`policy.onnx`, `manifest.json`, README to the Hub; only constant-command
episodic/perpetual policies are publishable).

---

## 8. Task-by-task walkthrough

Every task below uses **the same PPO configuration** (§4.1) unless stated. So for
each task the interesting questions are: *what is the MDP, what is rewarded,
what is curriculum'd, and what failure modes shaped the design.*

Legend for reward tables: **(+)** positive-weight reward, **(−)** cost.
"Self-negating" terms show a positive weight by design (§5.2).

### 8.1 Velocity — `Mjlab-Velocity-{Flat,Rough}-MicroDuck` (the main task)

*File: `tasks/microduck_velocity_env_cfg.py`. Also the shared base for other envs.*

- **Goal:** track velocity commands `vx ∈ ±0.4`, `vy ∈ ±0.3`, `ωz ∈ ±1.0` while
  following a head-pose command.
- **Robot model:** `walk` (feet-only collisions).
- **Command:** twist with 2% → 25% standing envs (curriculum) and a 15% turn-in-place
  bucket; head_pose ranges curriculum'd 5% → 100% of each joint's reachable range;
  body_pose kept tiny and its reward at 0 (slot stays alive).
- **Rewards:** see §5.1.
- **Terminations:** time-out and `nan_state`, plus whatever the mjlab template defines **(mjlab template, not read)**.
- **Curricula:** `action_rate_weight`, `standing_envs`, `head_pose_range`,
  `head_pose_bias_weight`, `com_range`, `head_com_range`, `body_pose_range`.
- **Design lessons:**
  - Ranges are *fixed*: a widening curriculum outpaced the robot and reward
    declined after iter 1000.
  - `upright` was strengthened to 2.0 / std² 0.05 because the walker leaned
    forward 2–4° and most falls were forward.
  - Tightening head std made the policy stand still (see §5.2); replaced by the EMA bias term.
  - CoM DR capped at ±15 mm: larger exceeded the foot support polygon and made
    backward balance untrainable.
- **Algorithm:** plain PPO, 50,000 max iterations, experiment `velocity`.

### 8.2 VelStand — `Mjlab-VelStand-{Flat,Rough}-MicroDuck`

*File: `tasks/microduck_velstand_env_cfg.py` (+ `tasks/distill.py`).*

- **Goal:** one policy that **walks, falls in a servo-protecting way, and gets back
  up**, so the daemon's "limp on fall" hack can be dropped. Motivated by real XL330
  gearboxes breaking.
- **Robot model:** `allcollisions` (full body can touch the floor; servo housings
  named for a dedicated impact sensor).
- **Structure:** velocity recipe *verbatim* as the walk layer, plus a recovery
  layer that is gated on being fallen (tilt > 40°) and contributes **zero while
  walking**.
- **Rewards added:**

  | Term | Function | Style |
  |---|---|---|
  | `upright_progress` (+5) | Δcos(tilt) | potential-based |
  | `height_progress` (+30) | Δ trunk z, capped 0.115 | potential-based |
  | `recovery_success` | bounty when tilt<25° and z>0.09 | ramped on later |
  | `fallen_tax` | hysteresis penalty until "recovered" | ramped on later |
  | `servo_impact`, `head_impact`, `trunk_impact` | `body_impact_cost` (force above 2 / 15 / 20 N) | cost (−) |
  | `servo_acc_spike` | joint accel > 300 rad/s² | cost (−) |
  | `servo_stall` | torque > 0.4 N·m at < 0.5 rad/s | cost (−) |
  | `gentle_rise` (+0.005) | trunk \|a_z\| | self-negating |
  | `action_rate_l2_fallen_scaled` | smoothness ×0.1 while fallen | avoids attempt tax |
  | `air_time` (`feet_air_time_upright`) | air time zeroed when fallen | stops "shaking a leg on the floor" |
- **Events:** a second `topple_push` event (±0.6 → ±1.2 m/s) so falls are frequent
  on-policy data; `random_prone_init` spawns face-down / face-up / **side** (side
  falls are 79% of real falls) with a ramp; a crouch-slice reverse curriculum.
- **Terminations:** `fell_over` (70°) only in the first phase, then disabled so a
  fall becomes a recovery opportunity; `fallen_too_long` (8 s) recycles hopeless episodes.
- **Algorithm:** **PPO + expert behaviour cloning** (`PpoWithExpertBc`), warm-started
  from the deployed walking checkpoint. Stand-up expert supervises fallen frames
  (tilt > 35°), the frozen walk checkpoint anchors upright frames (tilt < 25°), PPO
  alone owns the band in between and everything the rewards add.
- **The run log in the docstring is a masterclass** (runs 1–7): lie-still collapse
  from an attempt tax, spawn holes, BC destroying the walk, crouch-endpoint parking,
  reward-hacking, a rollback, and a catastrophic terrain-spawn bug (absolute z
  spawns inside rough terrain). Read it top to bottom.

### 8.3 StandUp — `Mjlab-StandUp-{Flat,Rough}-MicroDuck`

*File: `tasks/microduck_standup_env_cfg.py`. Episode 6 s. Experiment `microduck_stand`.*

- **Goal:** start sitting / face-down / face-up (via `set_ground_state`) and rise
  gently to a stable stand, then hold it and track a body-pose command.
- **Single fixed target from t=0** (no waypoints), so the policy discovers its own
  path.
- **Rewards:**

  | Term | Weight | Role |
  |---|---|---|
  | `pose_stand_legs` | +2.0 | Gaussian match to home (`pose_target_match`) |
  | `pose_stand_l1` | +1.25 | L1 gradient near the goal (`pose_l1_penalty`, self-negating) |
  | `height_stand` / `height_stand_sharp` | +1.0 / +1.0 | trunk z near `STAND_Z=0.115` |
  | `height_stand_l1` | +7.5 | far-field height gradient |
  | `com_upward_velocity` | +0.75 | dense "going up" signal (capped at z=0.125) |
  | `upright_linear` / `upright_sharp` | +1.5 / +1.5 | two-layer upright |
  | `standing_composite` | +3.75 | product of height·upright·pose |
  | `head_pose_tracking` | +0.75 | head commandable |
  | `gentle_rise` | +0.005 | \|a_z\| impact penalty (self-negating) |
  | `body_pose_tracking` | 0 → curriculum | added at iter ~2500 |
  | `action_rate_l2`, `joint_torque_rate`, `arrival_damping` | curriculum from ~0 | smoothness introduced late |
- **Curricula:** `ground_state_mix` (face-down/face-up share ramps until ~iter 2500),
  smoothness weights from ~3000, `standing_composite`/`upright_sharp` weights, push
  magnitude 0 → ±0.3.
- **Lesson encoded in comments:** any attempt tax active during recovery discovery
  prevented flips from ever being found in two runs; the fix was timing, not magnitude.

### 8.4 SitStand — `Mjlab-SitStand-{Flat,Rough}-MicroDuck`

*File: `tasks/microduck_sitstand_env_cfg.py`. Episode 12 s.*

- **Goal:** one policy, both directions, commanded by a flag in the twist slot
  (`[sit_flag,0,0]`, 0 = stand). The flag flips every few seconds mid-episode, so
  each episode contains descent, seated rest, rise, standing rest.
- **Trick:** the *target* (sit or stand keyframe and height) is **selected per env
  from the live command** by the `posture_*` reward family, while the reward
  structure is the standup recipe.
- **Rewards:** `posture_pose_match` (+4), `posture_pose_l1` (+1), two
  `posture_height_gaussian` (+1,+1), `posture_height_l1` (+6), `posture_rise_bootstrap`
  (+0.75), `posture_stillness` (+2), `posture_composite` (+3, includes a head factor),
  `upright_linear` (+2.5), `upright_while_tall` (+1.5), `head_pose_tracking` (+0.75).
- **Gentleness:** `descent_speed` (`trunk_downward_velocity_penalty`, +10 from step 0)
  and a mirrored `rise_speed` introduced by curriculum **after** the rise is
  discovered; `gentle_motion` on \|a_z\|.
- **Lessons:** the old sit keyframe tipped over (always check *tilt*, not just height);
  pushes early made the policy unlearn sitting (delayed push ramp); a seated-contact
  NaN fix (`nconmax=200`, solver iterations 30/50).

### 8.5 GroundPick — `Mjlab-GroundPick-{Flat,Rough}-MicroDuck`

*File: `tasks/microduck_ground_pick_env_cfg.py`.*

- **Goal:** crouch to bring the mouth tip **as close to the floor as possible without
  touching it**, then return to a clean stand. 4 s cycle (2 s down, 2 s up).
- **Phase clock in the command slot** `[cos 2πφ, sin 2πφ, 0]` (the policy needs to
  know *where it is in the cycle* because the objective flips from "go down" to "go up").
  Phase is randomised per env at reset to decorrelate envs.
- **Rewards:**

  | Term | Weight | Role |
  |---|---|---|
  | `mouth_ground_proximity` | +3 | Gaussian on mouth height, gated by phase (descent + hold) |
  | `mouth_perpendicular_to_ground` | +2 | mouth points down |
  | `ground_pick_return_pose_{legs,neck}` | +6, +6 | return pose in the up phase |
  | `return_upright` | +4 | upright on return |
  | `feet_grounded` / `feet_flat` | +3 / −2 | feet stay planted |
  | `head_impact_penalty` | −2 | *strong*: forbids contact, so equilibrium = just above floor |
  | `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2` | −2, −1, −5e-3 | heavier than walking: slow careful reaching |
  | `mouth_payload_force` | 0 | optional payload-DR event |
- **Concept to notice:** the task is specified in **task space** (mouth height above
  the floor), not as a target joint pose; the policy finds the crouch itself.

### 8.6 BallKick — `Mjlab-BallKick-Flat-MicroDuck`

*File: `tasks/microduck_ball_kick_env_cfg.py`. Episode 5 s. Experiment `ball_kick_<foot>`.*

- **Goal:** from a stand, kick a 70 mm / 15 g ball forward with the right foot
  (`KICK_FOOT="right"`), at a target speed of 1.0 m/s, without falling; then stand.
- **Blind actor, seeing critic:** no ball in the actor obs (real robot has no ball
  sensor). Robustness to aim error comes from ±1.5 cm ball position DR. The critic
  gets ball position/velocity (`ball_pos_in_base`, `ball_vel_in_base`).
- **No phase command:** the kick pays from t=0, so an earlier kick collects more
  reward, hence the policy kicks immediately.
- **Rewards:**

  | Term | Weight | Role |
  |---|---|---|
  | `ball_forward_velocity` | +12 | *linear* in forward ball speed, clamped at target |
  | `ball_speed_overshoot_penalty` | −4 | makes the target speed the true optimum (a harder kick would otherwise keep paying) |
  | `support_foot_grounded` | +2 | left foot must stay down, so the right leg does the kick and hopping is prevented |
  | `pose_stand_legs/neck`, `height_stand` | +2 / +1 / +1 | settle back to a clean stand |
  | `upright` | +2 | |
- **Concepts:** asymmetric actor–critic; encoding *which leg* through geometry (ball
  spawn) plus an always-on support reward; a saturating reward alone does not remove
  "harder is better", so an explicit overshoot cost is needed.

### 8.7 Roulade — `Mjlab-Roulade-Flat-MicroDuck`

*File: `tasks/microduck_roulade_env_cfg.py`. Episode 5 s. Docstring records 5 runs.*

- **Goal:** forward roll over the flat head top and land on the feet.
- **Reward hacking history:** run 1 learned a ballistic "breakdance whip" (same 2π
  rotation, faster, no cost). Fixes now built in:
  - **`roulade_progress` (+8):** pays *increments of the max-so-far cumulative
    rotation* (potential-style, up to 2π). The rotation accumulator is
    **support-gated** (only counts while touching the ground); paid rate capped
    at 3 rad/s (faster forfeits the excess).
  - `roulade_overspeed` (−0.1): tax on |ω| > 4 rad/s. Note the physical calibration:
    a 25 cm robot naturally tumbles at 3.5–5.5 rad/s, so don't impose human-scale caps.
  - `roulade_head_pivot` (+0.5): head-ground contact *while rotating* in the
    30°–240° window, weighted by "flat top down" (teaches the chin tuck).
  - **Landing rewards gated on roll completion** (rotation frontier ≥ ~260°):
    `landing_composite` (+4), `upright_after_roll` (+1.5), `height_after_roll` (+1),
    `landing_sharp` (+2). "Do nothing" earns nothing; standing spawns can't farm them.
  - `roulade_stand_tax` (+5), `rise_velocity` (+0.75), `sagittal`/`lateral_vel`/`flatness`
    penalties keep the roll a *sagittal, flat* roll rather than a shoulder-roll.
  - Motion-blockers (`body_ang_vel` −0.002, `angular_momentum` −0.001) kept near zero.
- **Curriculum:** `roulade_spawn_mix` (reverse curriculum: a slice of episodes starts
  50°–185° into the roll, tucked, with forward angular momentum). The second half of
  a roulade *is* the face-up recovery problem.

### 8.8 Roller family (passive wheels under the feet)

Common: robot `groundcontact_rollers` (4 passive wheels, joints interleaved so lookup
by name), `entropy_coef=0.03` for velocity rollers, wheel-bearing friction DR plus a
`wheel_friction` curriculum.

#### 8.8.1 Roller velocity — `Mjlab-Velocity-Flat-MicroDuck-Rollers`
*`microduck_velocity_rollers_env_cfg.py`. Experiment `velocity_rollers`.*

- `cmd_x` semantics: **0 = coast, > 0 = push, < 0 = brake**; `cmd[2]` = heading error
  via `RelativeHeadingVelocityCommand`.
- Sole main positive reward is `wheel_speed` (+10): the robot must actually spin
  its wheels. Style-shaping rewards: `braking` (+1), `skating_air_time` (+1.5),
  `glide` (+4), `single_support` (+3), `forward_lean` (+1.5), `heading_hold` (+1),
  `com_height_target` (+2), `pose` (2), `upright` (2); costs: `gait_symmetry` (−1),
  `hip_roll_neutral` (−2), `feet_flat` (−2), `action_over_limit` (−0.5),
  `joint_torques_l2` (−1e-3), neck terms.
- Naturally converges to a swizzle-like motion but the alternating stride transfers
  poorly to the real robot.

#### 8.8.2 Swizzle — `Mjlab-Velocity-Swizzle-MicroDuck`
*`microduck_velocity_swizzle_env_cfg.py` (~200 lines).*

- Reuses the roller env wholesale (robot, obs, DR, curricula) and **only swaps the
  reward recipe**: remove the anti-swizzle / stride terms, add `leg_symmetry` (+2,
  legs mirror) and `grounded` (+1, both blades down); `heading_tracking` and
  `head_pose_tracking` set to 0.
- A model of the "change the reward, keep everything else" workflow.

#### 8.8.3 Roller crouch — `Mjlab-RollerCrouch-Flat-MicroDuck`
*`microduck_roller_crouch_env_cfg.py`.*

- One-shot gesture: crouch and glide on the momentum (about a 1 s plateau), then
  stand up and hand control back to the roller policy.
- Hybrid: roller physics + ground-pick phase machinery (`GroundPickPhaseCommand`, 4 s).
- Height target is a **trapezoid** vs phase (up → down → plateau → up)
  via `crouch_glide_height_by_phase`.
- Rewards: `crouch_glide_pose` (+6), `crouch_glide_pose_l1` (+2), `forward_speed` (+1),
  `crouch_forward_lean` (+1), `upright` (2), `feet_flat` (−2), smoothness.

#### 8.8.4 Roller slope — `Mjlab-RollerSlope-Flat-MicroDuck`
*`microduck_roller_slope_env_cfg.py`. 8000 iterations.*

- Robot spawns on flat ground with a forward impulse, rolls down a ramp and has to
  **stay upright without any steering** (twist command neutralised,
  `rel_standing_envs=1.0`).
- Custom terrain `FlatRampTerrainCfg` with a slope-steepness curriculum
  (`terrain_levels_slope`).
- Small reward stack: `upright` Gaussian (+3), `alive` (+1), `heading_hold` (+1.5), `feet_flat` (−2),
  neck terms, `joint_torques_l2`. Terminations: `fell_over`, `fell_into_void`, `nan_state`.
- Demonstrates that a *survival-style* task can be very small.

#### 8.8.5 Roller stand-up — `Mjlab-RollerStandUp-Flat-MicroDuck`
*`microduck_roller_standup_env_cfg.py`. Episode 6 s.*

- Port of the StandUp recipe to the roller model: start face-down / face-up / already
  standing, rise onto the wheels and hold.
- Same reward *shape* as StandUp (pose, height, upright, composite, com_upward_velocity)
  but **much larger weights** (composite +15, height L1 +30, pose +8), and `joint_torque_rate_l2` (−0.2).
- The genuinely new idea is an **inverted friction curriculum**: because the wheels
  roll, there is no grip to push against, so the run starts with nearly *braked*
  wheels and ramps to real rolling friction. If `standing_composite` collapses at a
  stage, the "feet grip the floor" gesture doesn't transfer and a skater technique
  (knee push, one skate at a time) has to be guided.

#### 8.8.6 Spin — `Mjlab-Spin-Flat-MicroDuck`
*`microduck_spin_env_cfg.py`.*

- One cyclic gesture: about one counter-clockwise turn at ~3 rad/s then stop cleanly.
- The phase drives a **target yaw rate** (an *outcome objective*), not a joint pose.
- Rewards: `spin_rate_track` (+6), `spin_rate_l1` (+0.5), `spin_stay_in_place` (−3),
  `spin_wheel_differential` (+1, left blade backward / right forward), `leg_antisymmetry`
  (+1, decaying by curriculum), `spin_grounded` (+0.5), `feet_flat` (−2), `upright` (2).
- The two decaying "primers" (`spin_wheel_differential`, `leg_antisymmetry`) push toward
  differential rolling, the only mechanism that certainly works with 4 passive wheels.

---

## 9. Task comparison table

| Task | Model | Episode | Command in twist slot | Main positive signal | Distinctive trick | Algorithm |
|---|---|---|---|---|---|---|
| Velocity | walk | 20 s | `[vx,vy,ωz]` | velocity tracking + air_time | head-bias EMA penalty, fixed ranges | PPO |
| VelStand | allcollisions | 20 s | `[vx,vy,ωz]` | velocity + Δupright/Δheight | servo-protection costs, topple pushes | **PPO + expert BC**, warm start |
| StandUp | groundcontact | 6 s | ~0 (body_pose used) | composite score, height | ground_state_mix, late smoothness | PPO |
| SitStand | groundcontact | 12 s | `[sit_flag,0,0]` | posture-conditioned composite | target chosen from the live command | PPO |
| GroundPick | groundcontact | phase cycle (4 s) | `[cos,sin,0]` | mouth proximity (phased) | task-space target, head-impact veto | PPO |
| BallKick | groundcontact | 5 s | 0 | ball forward speed | blind actor / seeing critic, overshoot cost | PPO (asym. critic) |
| Roulade | groundcontact | 5 s | 0 | progress on support-gated rotation | completion-gated landing, reverse curriculum | PPO |
| Roller velocity | rollers | 20 s | coast/push/brake | wheel_speed | style rewards | PPO, entropy 0.03 |
| Swizzle | rollers | 20 s | same | wheel_speed + symmetry | reward-only swap | PPO, entropy 0.03 |
| Roller crouch | rollers | phase cycle | `[cos,sin,0]` | pose by phase | trapezoid height target | PPO |
| Roller slope | rollers | – | 0 | upright + alive | slope terrain curriculum | PPO |
| Roller standup | rollers | 6 s | 0 | composite | inverted wheel-friction curriculum | PPO |
| Spin | rollers | phase cycle | `[cos,sin,0]` | yaw-rate tracking | outcome objective + decaying primers | PPO |

(20 s is the mjlab velocity template default; check the template if you need the exact number.)

---

## 10. How to build and debug a new task

Following `AGENTS.md`:

1. **Pick the closest template** and build on it: locomotion → velocity; ends in a
   pose → standup; commanded two-state → sitstand; dynamic maneuver → roulade.
   `make_microduck_velocity*_env_cfg` gives you DR, obs noise, delays and NaN guards
   for free. A standalone env has to port all of that (`_safe` critic terms,
   `nan_state`, `expand_bam_friction_fields`, encoder bias, IMU misalignment).
2. **Check physics before training.** A target pose must be a stable equilibrium:
   hold its ctrl for 3 s from noisy starts and check *tilt*, not just height. Measure
   target heights from the actual robot in sim.
3. **Follow the config conventions:** `ENABLE_*` toggles and tuned constants at the top,
   a factory `make_..._env_cfg(play, rough)`, register it in `tasks/__init__.py`, a
   dedicated `RslRl...RunnerCfg` with its own `experiment_name`.
4. **Write cfg tests** (`tests/test_*_cfg.py`): joint indices resolve on the real model,
   reward signs are right, gates open/closed where expected.
5. **Smoke test** 64 envs × 5 iterations: builds, NaN-free, obs is 61-D, every reward
   term computes, ONNX exports.
6. **Train and expect 2–5 rounds of reward-hacking whack-a-mole.** Then
   **measure before theorising**: run a headless eval of the checkpoint (per-spawn-type
   batteries, end-state clusters) before touching rewards. Past "failures" turned out to
   be early checkpoints and success criteria that split one behaviour in half.

A reusable debugging checklist:

- Reward rises, behaviour bad → a term is being farmed; check each positive term against
  every stable flop pose.
- Behaviour "does nothing" → an attempt tax is active during discovery, or a gradient is
  invisible (std too tight / gate too hard).
- Metric drops exactly at a curriculum boundary → stage introduced too early.
- Works in viewer, fails on robot → normaliser not baked, unmodelled delay/noise, or the
  policy exploits sim-only information.

---

## 11. Suggested study path

1. **RL basics you need**: MDP, return, policy gradient, PPO clipping, GAE.
   Then read §4 here and match each hyper-parameter to a concept.
2. **`microduck_velocity_env_cfg.py` top to bottom.** It shows the whole manager
   API: rewards, events, observations, commands, curricula, terrain.
3. **`tasks/mdp.py`**: read `head_pose_tracking`, `reward_weight`, `upright_progress`,
   `height_progress`, `standing_composite_score`. Those five explain most patterns.
4. **StandUp → SitStand**: single-target rewards, composites, attempt-tax timing.
5. **Roulade** (docstring first, then `roulade_progress`): the best example of
   closing reward-hacking loopholes with state gates.
6. **BallKick**: asymmetric actor–critic and reward design around a saturating target.
7. **VelStand + `distill.py`**: when RL alone cannot discover a skill, and how to combine
   RL with behaviour cloning safely.
8. **Roller tasks**: how much of a new task is *reused* (swizzle is ~200 lines).
9. **`tests/`**: what invariants the authors chose to lock in.
10. **Experiments to try:** run the smoke test; train Velocity for ~500 iterations and
    watch `Episode_Reward/*`; flip `ENABLE_SYMMETRY`; change `head_pose_tracking` std and
    reproduce the "policy stands still" failure; try a wrong-sign penalty and watch it get farmed.

Further reading in the repo: `AGENTS.md` (distilled playbook),
`docs/superpowers/specs/*` and `plans/*` (design docs per roller task),
`docs/roller_standup_policy_summary.md`, `scripts/hf/README.md` (HF Jobs).
