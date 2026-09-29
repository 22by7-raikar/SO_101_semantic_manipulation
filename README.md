# SO_101_semantic_manipulation

Minimal research repository for language-grounded robotic manipulation with an SO-101 leader–follower arm and a fixed Intel RealSense RGB-D camera.

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
