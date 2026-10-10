# Predictive guidance for VLAs on SO-101 — project plan

Live, shared copy (with diagrams and comments): https://claude.ai/code/artifact/f4ab0057-29e5-419a-802b-4f1b17aa6bd5
This file is a snapshot from Oct 10, 2026. Edit either, but keep them in step.

## Goal

In 6 weeks (Oct 12 – Nov 20, 2026), make a fine-tuned SmolVLA on the SO-101 avoid static and moving obstacles it never saw in training, by steering its flow-matching sampler with planning costs and backing it with a CBF-QP safety filter.

**Research question:** can training-free, prediction-aware cost guidance inside a VLA's action sampler cut collisions under distribution shift without hurting task success, and how does it compare to post-hoc filtering?

**Contributions:**

1. Inference-time cost guidance for a flow-matching VLA, using a distance field from live RealSense depth (clearance, joint limits, smoothness).
2. Predictive guidance: forecast positions of a moving obstacle folded into a time-indexed cost.
3. Guidance vs. CBF-QP filter vs. both, on a real low-cost arm.
4. Stretch: distill guided rollouts back into the policy.

**Target:** 4-page workshop paper / arXiv preprint, project page with videos, open dataset and code.

## System

```
RealSense depth → arm/target masking → track + forecast → distance field + predicted occupancy
                                                               │                  │
                                         cost gradients at each flow step     clearance
                                                               ▼                  ▼
policy inputs (wrist + scene RGB, joints) → SmolVLA flow sampler (N chunks) → select + spread → CBF-QP → SO-101
        ▲                                                                                             │
        └───────────────────────────────── joint state + new frames ──────────────────────────────────┘
```

## Compute: everything on the 6 GB RTX 3060

| Component | Runs on | Rough VRAM (estimate; measure in Week 1) |
| --- | --- | --- |
| SmolVLA inference (~450M, fp16), N samples batched | GPU | ~1.5–3 GB depending on N |
| Guidance: FK + cost gradient on the predicted chunk | GPU or CPU | < 100 MB |
| Distance field (voxel grid + EDT) | CPU | 0 |
| Arm masking (collision spheres projected into depth) | CPU | 0 |
| Tracking + forecast (clustering + Kalman) | CPU | 0 |
| CBF-QP filter (OSQP) | CPU | 0 |

Guidance never backpropagates through the VLA. Training and evaluation never run at the same time. Training: vision encoder frozen, mixed precision, small batch + gradient accumulation; Kaggle/Colab if it runs out of memory. π0.5 / GR00T are out of scope.

## Data collection (Weeks 1–2)

~60 clean-scene teleop demos of **one task: pick a block and place it in a bowl**: the first ~20 by the end of Week 1, the remaining ~40 in Week 2. Stacking is the follow-on task once core results are in.

Each frame: wrist RGB, RealSense RGB, RealSense depth (offline perception only), 6 joint positions (state), leader joint targets (action).

| Axis | Levels | How |
| --- | --- | --- |
| Block start position | 12 taped spots (4 × 3 grid); 2 held out | Rotated by the schedule |
| Bowl position | 4 spots; 1 held out | Rotated by the schedule |
| Block orientation | 0°, 45°, 90° | Cycled per episode |
| Block | 3 colours, one larger | Instruction names it: "put the red block in the bowl" |
| Lighting | 2–3 settings | Changed between sessions |
| Table surface | 2 | Half the episodes each |
| Operator motion | Left / right / above approach; slow and normal | Noted in the log |
| Recoveries | ~10% of episodes | Miss the grasp slightly, then correct |

Kept fixed: no obstacles or distractors; RealSense pose (recalibrate if bumped); wrist mount; home pose at episode start.

Process: generate a balanced episode-schedule CSV first → record in sessions of ~15 → review in LeRobot's dataset visualizer and re-record bad demos → keep the CSV as metadata for seen vs. held-out splits. Augment with colour jitter and small crops; keep geometric augmentation mild. Stay with one task at least until Gate 1 passes.

## Evaluation

Task: pick a block and place it in a bowl. Demos have no obstacles, so every test-time obstacle is out of distribution. Each condition runs only the methods that answer a question about it.

| Method | What it is |
| --- | --- |
| VLA | Fine-tuned SmolVLA, raw |
| VLA + QP | Post-hoc CBF-QP filter |
| VLA + guidance | Cost guidance in the sampler, current-frame obstacles |
| VLA + predictive guidance | Guidance with forecast occupancy |
| Full | Predictive guidance + QP |
| Distilled (stretch) | Fine-tuned on guided rollouts, run without guidance; only if time remains |

Trial matrix (20 trials per method in each condition, marked start positions):

| Condition | Methods run | Trials |
| --- | --- | --- |
| Clean (seen + held-out positions) | VLA, VLA + guidance, Full | 60 |
| Static obstacle (box or mug at 3 spots) | VLA, VLA + QP, VLA + guidance, Full | 80 |
| Moving obstacle (repeatable rig) | VLA, VLA + QP, VLA + guidance, VLA + predictive guidance, Full | 100 |
| Clutter (3–4 distractors) | VLA, VLA + guidance, Full | 60 |
| Ablations (static obstacle, 10 each) | 4 sweeps | 40 |
| **Total** | | **340** |

About 11 hours of robot time at ~2 minutes per trial with reset. Predictive guidance equals plain guidance when nothing moves, so it runs only in the moving condition.

**Collision:** the robot touches or moves a non-target object, judged frame by frame from video by both of us, defined in the evaluation protocol before Week 2 trials. Minimum clearance from the distance field is the objective backup to the video judgement.

Metrics: success rate, collision rate, min clearance (cm), mean jerk, completion time, sampler latency (ms/chunk). Ablations: guidance strength, guided flow steps (all vs. late), forecast horizon, N samples. Uncertainty: AUROC of sample spread vs. failure.

## Gates

- **Gate 1 · Oct 23 (may slip to Mon Oct 26 if fine-tuning runs long):** SmolVLA ≥ 50% on Clean, else add demos and narrow variation before moving on.
- **Gate 2 · Nov 6:** guided loop runs live with a moving obstacle.
- **Gate 3 · Nov 13:** all trials logged with video.

## Risks

| Risk | Fallback |
| --- | --- |
| Base VLA too weak | More demos; narrower positions; retrain on Kaggle/Colab with a larger batch |
| Guidance pushes actions off-distribution | Guide only late flow steps; cap λ; lean on QP and report the trade-off |
| Latency too high | Fewer samples / flow steps; guide every k-th chunk; lower image resolution |
| Noisy depth near arm / small objects | Voxel filter, temporal smoothing, bigger margins and obstacles |
| Hand forecasting unreliable | Constant-velocity model; scripted mover for repeatable trials |
| Prior work already does VLA cost guidance (Week 1 literature check, start with RTC) | Emphasise the predictive part and the real-robot comparison |
| Hardware issues (servo overheating, calibration drift) | Recalibrate weekly; keep sessions short; spare servo on hand |
| Safety | Software workspace/torque limits; hand on e-stop every rollout |

Closest prior work: real-time chunking (RTC), which LeRobot already implements for SmolVLA, guides a flow policy's sampler toward consistency with the previous chunk. This project guides with geometric and predictive planning costs instead.

---

# Checklist — motion planning (MP) and shared (Both)

Teammate's CV tasks are in the live doc.

## Week 1 · Oct 12–16 — setup, robot model, calibration

Done when: a teleop episode records with both cameras, FK matches the real gripper to ~1 cm, and the RealSense cloud lines up with the arm.

**Setup (Both)**
- [x] `bash scripts/setup.sh robot` in your own clone (Python 3.11, LeRobot 0.4.4, locked); `uv run --no-sync so101-doctor --gpu --hardware`; confirm `torch.cuda.is_available()`
- [x] Ports via `uv run --no-sync lerobot-find-port`; user in `dialout`; `/dev/serial/by-id/...` recorded in `configs/lab.local.yaml`
- [x] Calibrate follower and leader with `lerobot-calibrate`
- [x] Teleoperate 5 minutes with `lerobot-teleoperate`; note any servo running hot or jittering
- [ ] Identify cameras (`lerobot-find-cameras`); name them `wrist` and `scene`
- [ ] Stream aligned colour + depth at 640×480; save intrinsics and depth scale to `lab.local.yaml`
- [ ] Time a short SmolVLA fine-tune (~1k steps) on the laptop to size the Week 2 run
- [ ] (CV) Generate the episode schedule and record the first ~20 demos by Friday

**Robot model (MP)**
- [x] Get the SO-101 URDF (TheRobotStudio SO-ARM100 repo), pin its commit, load it with `yourdfpy`
- [x] Joint conversion LeRobot degrees / gripper 0–100 → URDF radians in `kinematics.py`, with a unit test (offsets are zero: the URDF zero is the middle of each calibrated range)
- [ ] Verify each joint's direction (`JOINT_SIGNS` in `kinematics.py`) on the real arm
- [ ] FK check: 5 poses, ruler vs. FK (target < 1 cm)
- [x] Joint limits in `kinematics.py`; servo datasheet maximum velocity (4.7 rad/s) as the hard limit
- [ ] Planning velocity limit per joint: 99th-percentile speed from teleop logs (the speeds the policy saw in demos)
- [x] Fit 5 collision spheres per link (`configs/so101_collision_spheres.yaml`, `scripts/fit_collision_spheres.py`): every mesh surface point covered with ≥ 0.76 mm to spare; spheres claim ~5–13 mm (mean) beyond each link
- [x] Self-collision checks on the spheres: adjacent pairs skipped, 3 pairs proven unable to touch skipped, 12 pairs checked; never misses a mesh contact
- [ ] View the spheres in meshcat over the mesh

**Camera-to-base calibration (MP)**
- [ ] Clamp the RealSense; mark its position
- [ ] 4–6 cm AprilTag fixed to the gripper
- [ ] 15–20 poses across the workspace: joint state + tag pose
- [ ] Solve eye-to-hand with OpenCV's hand-eye solver; store `T_base_camera` (`p_base = T @ p_camera`)
- [ ] Check FK gripper projected into the image on held-out poses (target 1–2 cm)

**Literature check (Both, half a day)**
- [ ] arXiv: inference-time guidance/steering of VLA, flow-matching or diffusion policies; collision-aware action chunking; start with RTC
- [ ] 5 closest papers, one line each on how this project differs

## Week 2 · Oct 19–23 — distance field, QP filter, baseline

Done when: the QP keeps replayed chunks clear of a test obstacle, and the Clean baseline has 20 logged trials (Gate 1). CV records the remaining ~40 demos and fine-tunes SmolVLA this week.

**Distance field (MP)**
- [ ] Interface with teammate: masked cloud in base frame, N×3 numpy + timestamp
- [ ] Workspace box, 1 cm voxels (`cost_field.voxel_size_m`)
- [ ] `scipy.ndimage.distance_transform_edt` on occupancy
- [ ] Torch trilinear query (`grid_sample`) → differentiable distance + gradient
- [ ] Test with a box at known points (1–2 cm)
- [ ] `clearance(q)`: FK spheres → min(distance − radius)

**CBF-QP filter (MP)**
- [ ] Formulation on paper: variables = chunk joint positions; objective = track VLA chunk + smoothness; constraints = joint and velocity limits, linearised discrete-time CBF on clearance
- [ ] Prototype in cvxpy, then OSQP (add both with locked pins)
- [ ] Offline test on recorded chunks with a synthetic obstacle; plot clearance before/after
- [ ] Time it (ms per chunk)

**Robot loop (MP)**
- [ ] observation → policy → optional filter → `send_action`, opt-in via config
- [ ] Software workspace box + keyboard e-stop that holds position

**Baseline (Both)**
- [ ] Tape seen/held-out positions; one-page evaluation protocol (start pose, reset, timeout, success; collision = the robot touches or moves a non-target object, judged frame by frame from video by both, with min clearance as the objective backup)
- [ ] 20 Clean trials with fine-tuned SmolVLA; log outcome + video
- [ ] Gate 1: ≥ 50% success, or add demos and narrow variation before Week 3 (may slip to Mon Oct 26 if fine-tuning runs long)

## Week 3 · Oct 26–30 — guidance inside the SmolVLA sampler

Done when: guided chunks bend around a static obstacle offline, and live trials show fewer collisions without a large Clean drop.

**Read the sampler (MP)**
- [x] SmolVLA's sampling loop (LeRobot 0.4.4): 10 steps, time runs from 1 to 0, update x ← x + dt·v with dt = −0.1; "late steps" means t near 0
- [ ] How actions are normalised (dataset stats), to undo it in torch

**Add the hook (MP)**
- [ ] Wrap `denoise_step` the way LeRobot's real-time chunking (RTC) code does (`modeling_smolvla.py:847`); make it a `guide(x_t, v_t, t) → v_t` hook, no-op by default
- [ ] Re-enable gradients inside the hook with `torch.enable_grad()`, since `sample_actions` runs under `torch.no_grad()`
- [ ] Outputs identical with the hook off (same seed)
- [ ] Predicted final chunk: x̂ = x_t − t·v_t
- [ ] Differentiable unnormalisation to joint positions
- [ ] J = softplus(margin − clearance) + joint-limit penalty + second-difference smoothness
- [ ] ∇J w.r.t. x_t with v_t detached (no backprop through the VLA); v_t ← v_t − λ(t)·∇J
- [ ] Clip gradient norm; λ on all steps vs. late steps only (t near 0)

**Offline test (MP)**
- [ ] Fake obstacle on a recorded demo path; plot guided vs. unguided chunks in meshcat
- [ ] Sweep λ; find where chunks stop looking like demos

**Samples and timing (MP, with CV)**
- [ ] Batch N samples; pick lowest cost; log spread
- [ ] ms per chunk for N = 1, 4, 8

**Live trials (Both)**
- [ ] 10 Clean trials with guidance on
- [ ] Static obstacle at 3 spots: VLA vs. +QP vs. +guidance, 5 trials each

## Week 4 · Nov 2–6 — predictive costs for moving obstacles

Done when: the full guided loop runs live while an object crosses the workspace (Gate 2).

- [ ] Agree tracker output: per obstacle position, velocity, radius, covariance, timestamp (base frame)
- [ ] For chunk step k, predict obstacle at position + velocity × (t_k − t_measurement)
- [ ] Inflate radius with forecast uncertainty and with k
- [ ] Combine with the static field (smaller clearance wins)
- [ ] Same time-indexed constraint in the QP
- [ ] Execute fewer steps per chunk and re-sample sooner while an obstacle moves
- [ ] Blend overlapping chunks (try LeRobot's RTC first; it may already do this); check jerk in logs
- [ ] (Both) Repeatable mover (string or toy car), marked start/stop; confirm roughly constant speed
- [ ] 10 trials: current-frame vs. predictive cost
- [ ] Gate 2

## Week 5 · Nov 9–13 — experiments

Done when: every trial in the matrix is logged with video (Gate 3).

- [ ] Freeze code and parameters; tag `eval-v1`
- [ ] Randomised trial order across methods and conditions
- [ ] Per-trial log: method, condition, position id, success, collision, min clearance, jerk, time, video
- [ ] (Both) Matrix: 300 trials across Clean, static, moving, clutter (see Evaluation)
- [ ] Ablations on the static-obstacle condition, 10 trials each: λ, guided steps, forecast horizon, N (40 trials)
- [ ] Metrics per method × condition with bootstrap CIs; clearance-over-time plots
- [ ] Gate 3

## Week 6 · Nov 16–20 — write-up and release

- [ ] Pick venue; check deadline and page limit
- [ ] Figures: architecture, results table, clearance plot, moving-obstacle frames
- [ ] MP: Method and Experiments/Ablations sections
- [ ] Both: Intro, Related work, Limitations; full read-through each
- [ ] README with setup + one-command evaluation; dataset + checkpoint on the HF Hub; release tag
- [ ] 2-minute video; resume bullet with the headline number
- [ ] Plan the stacking extension
