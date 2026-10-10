# SO-101 follower model (vendored)

`so101_new_calib.urdf` is copied unmodified from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) at commit
`a758567c3978dfeefe282ede0500085a48fe8f78` (`Simulation/SO101/so101_new_calib.urdf`),
licensed Apache-2.0 (see `LICENSE`).

| File | SHA-256 |
| --- | --- |
| `so101_new_calib.urdf` | `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c` |

"New calibration" means each joint's zero is the middle of its range, matching LeRobot's
degree normalization (`0° = (range_min + range_max) / 2`). Joint directions and the gripper
mapping must still be checked against the real arm; see `docs/HARDWARE_SETUP.md`.

Meshes (~16 MB) are not committed. Fetch and verify them when needed for visualization or
collision checking:

```bash
uv run --no-sync python scripts/fetch_so101_meshes.py
```

Forward kinematics (`so101_semantic.kinematics`) does not need the meshes.
