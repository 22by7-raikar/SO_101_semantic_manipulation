# SO_101_semantic_manipulation

Minimal research repository for language-grounded robotic manipulation with an SO-101 leader–follower arm and a fixed Intel RealSense RGB-D camera.

## Quick start

Install the pinned uv version and host prerequisites from [Environment setup](docs/ENVIRONMENT.md), then choose a profile in your own clone:

```bash
bash scripts/setup.sh core    # Apple Silicon Mac or Ubuntu: geometry/planning
# bash scripts/setup.sh vision  # Add PyTorch and vision model interfaces
# bash scripts/setup.sh robot   # Ubuntu: GPU, RealSense, Feetech and LeRobot
uv run --no-sync pytest -q
uv run --no-sync so101-demo
```

Python 3.11.15 and all resolved dependencies are recorded in `.python-version`, `pyproject.toml` and `uv.lock`. See [Starting the research together](docs/PROJECT_START.md) for the perception/planning interface and first milestones. The demo is synthetic and never moves the robot.

## Research question
Do language-conditioned semantic costs improve collision-checked motion planning when object placement and task instructions change?

## Hardware and development context
- **Robot:** SO-101 leader–follower arm
- **Perception:** Fixed Intel RealSense RGB-D camera
- **Primary development machine:** Ubuntu ASUS laptop with RTX 3060 (6 GB)
- **Collaboration:** Contributors may also edit from macOS

## Planned pipeline
1. Camera calibration and RGB-D perception
2. Object grounding and 3D geometry
3. Semantic cost representation
4. SO-101 kinematics and motion planning
5. Guarded hardware execution
6. Reproducible evaluation

## Proposed comparisons
- Geometry-only planning
- Manually labeled semantic costs
- Language-grounded waypoints
- ACT baseline
- SmolVLA baseline

## First milestone
Calibrated hardware teleoperation and a verified camera-to-robot transform.

## Remote GPU access

See [Shared Ubuntu GPU laptop setup](docs/REMOTE_GPU_SETUP.md) for two-person SSH access over Tailscale, separate user accounts, persistent sessions, and GPU coordination.
