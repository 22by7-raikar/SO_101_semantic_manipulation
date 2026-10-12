# CLAUDE.md

Context for Claude Code working in this repository. Read this first, then `docs/PLAN.md`.

## Project in one paragraph

Two-person, 6-week project (Oct 12 – Nov 20, 2026) on an SO-101 leader–follower arm. A SmolVLA policy is fine-tuned on clean-scene pick-and-place demos; at test time, planning costs built from live RealSense depth (clearance from a distance field, joint limits, smoothness, and forecast positions of moving obstacles) are injected as gradients into SmolVLA's flow-matching sampler, with a CBF-QP safety filter as backup. The research question: does training-free, prediction-aware cost guidance inside a VLA's sampler cut collisions under distribution shift without hurting task success, and how does it compare to post-hoc filtering? Full plan, schedule and checklists: `docs/PLAN.md`.

## People and ownership

- **Akshay — motion planning (MP):** URDF/FK, limits, collision spheres, camera-to-base calibration, distance field, CBF-QP filter, guidance hook in the SmolVLA sampler, predictive costs, execution loop, ablations.
- **Anuj — ML/CV:** RealSense pipeline, arm and target masking, obstacle tracking and forecasting, dataset collection, SmolVLA fine-tuning, uncertainty from sample spread, evaluation harness, distillation.
- Interface between them: point clouds / distance fields / obstacle tracks in the **robot base frame**, metres, with timestamps.

## Hardware and compute

- SO-101 follower + leader (6 joints incl. gripper); fixed Intel RealSense D435 (`scene`: RGB + depth); icSpring USB RGB camera on the gripper (`wrist`). Device ids, stream settings, factory intrinsics and depth-to-colour extrinsics are under `cameras:` in `configs/lab.local.yaml`.
- Camera settings: policy colour 640×480 @ 30 fps for both cameras, identical in demos and evaluation (SmolVLA resizes to 512×512). Distance-field depth 848×480 @ 30 fps, emitter on, spatial filter; temporal filter only for static scenes (it smears moving obstacles); no hole filling. Only one process can open a camera, so at evaluation our pipeline owns the RealSense and serves both. The wrist camera is policy input only and needs no geometric calibration.
- Follower: 6× Feetech STS3215 on a 12 V supply. All six read 11.8–11.9 V with a 14.0 V `Max_Voltage_Limit`, consistent with the 12 V C018 variant (30 kg·cm stall, 0.222 s/60° ≈ 4.7 rad/s no load); labels not checked. LeRobot caps the gripper at `Max_Torque_Limit` 500 (50%).
- Leader: STS3215s on a 5 V supply with mixed gearing (shoulder_lift reads an 8.0 V limit, the 7.4 V 1/345 part; the other five read 12.0 V). The leader is only read, never commanded.
- **Only GPU: RTX 3060 Laptop, 6 GB** (ASUS Zephyrus G14 GA401QM, Ubuntu). No cloud budget; Kaggle/Colab free GPUs are the fallback for training only.
- π0.5 / GR00T (3–4B params) do not fit and are out of scope. SmolVLA (~450M) is the only base policy.
- At evaluation, SmolVLA is the only real GPU consumer. Distance field, masking, tracking and the QP run on CPU. Guidance must **not** backpropagate through the VLA: compute the cost on the predicted final chunk with the velocity detached; only FK + cost are differentiated.
- One heavy GPU job at a time (the GPU is shared over SSH; see `docs/REMOTE_GPU_SETUP.md`).

## Environment (do not deviate)

- Python 3.11.15, uv 0.11.14, everything locked in `uv.lock`. LeRobot is pinned to **0.4.4** (`lerobot[feetech,smolvla]`), PyTorch 2.7.1 + CUDA 11.8.
- Setup: `bash scripts/setup.sh core|vision|robot` (robot = Ubuntu, GPU, RealSense, LeRobot).
- Always run with `uv run --no-sync ...` after choosing a profile; plain `uv run`/`uv sync` can strip optional extras.
- New dependencies (e.g. `osqp`, `cvxpy`, `pin`, `meshcat`): add to `pyproject.toml` with an exact pin, run `uv lock`, commit manifest + lock together. Never `pip install` into the env, never `uv lock --upgrade`.
- Before pushing: `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .`, `uv run --no-sync pytest -q`. CI runs the core profile on Ubuntu 22.04 and macOS 14 and cannot test hardware or CUDA.

## Code layout

- `src/so101_semantic/kinematics.py` — the one LeRobot → URDF conversion module. LeRobot degrees (gripper 0–100) → URDF radians; offsets are assumed zero (URDF zero = middle of each calibrated range). Both `JOINT_SIGNS` (all +1) and the zero-offset assumption are **unverified**; at the rest pose FK is off by ≥ 15 mm, consistent with a few degrees of offset. `SO101Model` gives FK (yourdfpy, no meshes needed) and `joint_limits(calibration_path)`: arm position limits = intersection of URDF and calibrated range; velocity/effort from the STS3215 datasheet (the URDF's `velocity="10" effort="10"` are upstream placeholders). Helpers: `limit_violations`, `clip_to_limits`, `min_segment_duration`.
- `src/so101_semantic/collision.py` — collision spheres: `load_spheres()` reads `configs/so101_collision_spheres.yaml` (5 per link, link frame, metres, 1 mm margin included); `SO101Model.collision_spheres(q)` places them in the base frame. Generated by `scripts/fit_collision_spheres.py` from the meshes; regenerate if the URDF changes (a test checks its SHA-256). Spheres overestimate the arm by ~5–13 mm on average, so clearance is conservative. `base_link` spheres extend below the table: exclude `base_link` from environment clearance and use it only for masking.
  Self-collision: `SO101Model.self_collision_distances(q)` / `self_collisions(q, margin)` check 12 of the 21 link pairs with spheres. Skipped (in the YAML with reasons): the 6 adjacent pairs, and 3 pairs two joints apart whose meshes the generator proves stay ≥ 2.5 mm apart over a full sweep of those joints (base/upper arm, lower arm/gripper, wrist/moving jaw). Never misses a mesh contact (tested); flags ~20% of random configurations vs ~15% where meshes come within 3 mm. Until joint offsets are verified, do not trust self-collision results near the folded rest pose: at the measured rest readings the model puts the gripper 15 mm into the table, which the real arm is not, and shows lower arm/shoulder contact that is likely the same model error.
- `src/so101_semantic/calib_sheet.py` — printed calibration sheet (4 tag36h11 tags ids 10–13 + 9 touch marks, A4 landscape): geometry, tag detection, sheet pose in camera (PnP), and `T_base_camera = T_base_sheet @ inv(T_camera_sheet)`. Used by `scripts/make_calib_sheet.py` and `scripts/calibrate_scene_camera.py`.
- `src/so101_semantic/fk_fit.py` — fits per-joint zero offsets (and checks joint directions) from touch points recorded with `scripts/record_touch_points.py`; the residual is the FK check. Results are not yet applied in `kinematics.py`.
- `src/so101_semantic/pose.py` — `so101-pose`: read-only follower joint readings + FK tool position. Reads registers only, leaves torque unchanged; do not run during teleop or calibration.
- `src/so101_semantic/distance_field.py` — `DistanceField.from_points(cloud, box_min, box_max, voxel)`: masked obstacle cloud (base frame) → occupancy → signed EDT (positive in free space; reads 0–1 voxel closer than the true surface, so conservative). `query(points)` numpy value + gradient, `.torch()` differentiable `grid_sample` query (same trilinear interpolation; out-of-box queries clamp to the border). `fill_down` fills below seen tops (unseen object sides). `clearance(model, q, field)` = min(distance − radius) over the FK spheres excluding `base_link`; `clearance_jacobian` gives per-sphere d/dq for the CBF. The cloud must exclude the arm and the grasp target; keep the table out and bound it separately.
- `src/so101_semantic/geometry.py` — `backproject(depth_m, K)` and `transform_points(points, T)`; strict validation, metres. Table handling: `fit_plane` (RANSAC + least squares, optional normal hint), `remove_table(points, plane, min_height=0.01)`, `plane_height`, `transform_plane`. The follower base, leader and camera stand all stand on the same table, which is z = −0.0024 m in the base frame (base mesh bottom); the depth-fitted plane mapped through `T_base_camera` must land there (calibration check). Keep the table out of the obstacle cloud and bound it as a separate floor constraint (no voxel error).
- `src/so101_semantic/demo.py` — synthetic cost-field writer (no devices). `so101-demo`.
- `src/so101_semantic/doctor.py` — read-only environment diagnostics. `so101-doctor [--vision --gpu --hardware --camera]`.
- `third_party/so101/` — `so101_new_calib.urdf` copied unmodified from TheRobotStudio/SO-ARM100 at a pinned commit (SHA-256 and Apache-2.0 license alongside). Meshes (~16 MB) are gitignored under `assets/`; fetch and checksum them with `scripts/fetch_so101_meshes.py`. Never edit the vendored URDF; keep overrides in our code or config.
- `scripts/` — `setup.sh core|vision|robot` (environment install), `fetch_so101_meshes.py`, `fit_collision_spheres.py`, `record_touch_points.py` (read-only touch recording for `fk_fit`), `make_calib_sheet.py`, `calibrate_scene_camera.py` (T_base_camera from the sheet + touch fit), `make_apriltags.py`, `stream_cameras.py` (live Rerun view of `scene`/`wrist` from `lab.local.yaml`; read-only), `realsense_fw_update.py` (dry run by default; `--flash` burns firmware).
- `docs/` — `PLAN.md` (plan + checklists), `CALIBRATION_STEPS.md` (operator checklist: sheet, touches, camera calibration), `HARDWARE_SETUP.md` (ports, udev, RealSense firmware, arm calibration, teleop, kinematic model check), `ENVIRONMENT.md`, `REMOTE_GPU_SETUP.md`.
- `configs/lab.example.yaml` → copy to `configs/lab.local.yaml` (gitignored) with measured intrinsics, `T_base_camera`, ports, URDF path. Null means "not measured yet"; never fill with guesses.
- `tests/` — pytest, must run without hardware.
- Gitignored: `data/`, `outputs/`, `checkpoints/`, `models/`, `calibration/local/`, weights, `.env`.

## Conventions

- Frames: `p_base = T_base_camera @ p_camera` (homogeneous column vectors). All geometry in metres, robot base frame, `xyz` axis order.
- LeRobot joint units are degrees (gripper 0–100); URDF uses radians. Keep the conversion (sign + offset per joint) in `kinematics.py` only.
- Code that touches motors must be explicit and opt-in (`execution.enabled: false` by default), with a software workspace box and an e-stop that holds position. Never run robot processes as root. Never add motor commands to diagnostics or tests.
- Record provenance with every result: git SHA, model revision, calibration revision, dataset version, seed.
- Pure-logic code (FK, distance field, QP, costs, forecasting) gets unit tests that run on CPU without hardware.

## Cost field contract v1 (NPZ)

Written by `so101-demo`; read with `numpy.load(path, allow_pickle=False)`.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer 1 |
| `costs` | float32 `[nx, ny, nz]`, nonnegative penalties |
| `observed` | bool, same shape; separates observations from unknown cells |
| `origin_m` | XYZ lower corner of the grid in `frame`, metres |
| `voxel_size_m` | Scalar cubic cell size, metres |
| `frame` | Frame id; the demo uses `synthetic_base`, real data a calibrated robot base |
| `axis_order` | `xyz`, not image row/column order |

- Cell `[i,j,k]` centre = `origin_m + voxel_size_m * ([i,j,k] + 0.5)`.
- An unobserved cell with zero cost is **unknown, not free**. Semantic cost is separate from occupancy and hard collision constraints.
- The demo marks only sampled surfaces as observed: no ray tracing, free-space carving, uncertainty or obstacle inflation.
- Before using a field in a differentiable planner, agree on interpolation and out-of-bounds behaviour (`torch.from_numpy` to load).
- Real records also need timestamps, camera intrinsics, depth units, calibration version and prompt/model provenance.

## Current status

See the checklists in `docs/PLAN.md`; the live, shared copy of the plan is the Claude Docs version linked at its top.
