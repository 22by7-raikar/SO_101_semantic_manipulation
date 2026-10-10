# Predictive guidance for VLAs on SO-101

A fine-tuned SmolVLA policy on an SO-101 arm, steered at inference time to avoid static and moving obstacles it never saw in training. Planning costs built from live RealSense depth are applied as gradients inside the policy's flow-matching sampler, with a CBF-QP safety filter as backup.

**Research question:** can training-free, prediction-aware cost guidance inside a VLA's action sampler cut collisions under distribution shift without hurting task success, and how does it compare to post-hoc filtering?

The full plan, schedule and checklists are in [docs/PLAN.md](docs/PLAN.md) (Oct 12 – Nov 20, 2026).

## Approach

1. **Policy:** SmolVLA (~450M parameters) fine-tuned on about 60 clean-scene teleop demos of one task: pick a block and place it in a bowl.
2. **Perception:** RealSense depth → arm and target masking → distance field in the robot base frame, plus tracking and forecasting of moving obstacles.
3. **Guidance:** at each flow step, estimate the final action chunk, compute a cost through forward kinematics (clearance, joint limits, smoothness, forecast occupancy) and push the sampler down its gradient. Guidance never backpropagates through the VLA.
4. **Safety filter:** a CBF-QP over each chunk enforces joint, velocity and clearance constraints.

| Method | What it is |
| --- | --- |
| VLA | Fine-tuned SmolVLA, raw |
| VLA + QP | Post-hoc CBF-QP filter |
| VLA + guidance | Cost guidance in the sampler, current-frame obstacles |
| VLA + predictive guidance | Guidance with forecast occupancy |
| Full | Predictive guidance + QP |

Evaluated on clean, static-obstacle, moving-obstacle and clutter scenes for success rate, collision rate, minimum clearance, jerk and sampler latency.

## Status

Working now, offline and tested:

- Pinned SO-101 URDF ([third_party/so101](third_party/so101/README.md)) with forward kinematics and LeRobot → URDF joint conversion.
- Joint position limits (URDF ∩ calibrated range) and servo velocity limits.
- Collision spheres for every link (5 per link, fitted to the meshes), placed with FK.
- Read-only follower pose readout (`so101-pose`).
- Depth back-projection and frame transforms; a synthetic cost-field writer (`so101-demo`).

Not yet built: camera-to-base calibration, distance field, QP filter, the guidance hook, and anything that commands motors.

## Quick start

Install the pinned uv version and host prerequisites from [Environment setup](docs/ENVIRONMENT.md), then choose a profile in your own clone:

```bash
bash scripts/setup.sh core    # Apple Silicon Mac or Ubuntu: geometry/planning
# bash scripts/setup.sh vision  # Add PyTorch and vision model interfaces
# bash scripts/setup.sh robot   # Ubuntu: GPU, RealSense, Feetech and LeRobot
uv run --no-sync pytest -q
uv run --no-sync so101-demo
```

Python 3.11.15 and all resolved dependencies are recorded in `.python-version`, `pyproject.toml` and `uv.lock`; LeRobot is pinned to 0.4.4. Always run with `uv run --no-sync`. Nothing in this repository moves the robot yet.

## Hardware

- **Robot:** SO-101 follower (6× STS3215, 12 V) and leader (5 V) for teleoperation
- **Cameras:** fixed Intel RealSense D435 (scene RGB + depth) and a USB RGB wrist camera
- **Compute:** one Ubuntu laptop with an RTX 3060 (6 GB), shared over SSH; SmolVLA is the only GPU consumer at evaluation

## Documentation

- [Project plan and checklists](docs/PLAN.md)
- [Environment setup](docs/ENVIRONMENT.md)
- [Hardware bring-up](docs/HARDWARE_SETUP.md): ports, RealSense firmware, arm calibration, teleoperation, kinematic model check
- [Shared Ubuntu GPU laptop setup](docs/REMOTE_GPU_SETUP.md): two-person SSH access over Tailscale and GPU coordination
