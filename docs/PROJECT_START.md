# Starting the research together

The initial target is the proposal's language-grounded 3D cost field, with independently testable perception and planning components. The synthetic demo is a development fixture, not the research system.

## Shared data contract (version 1)

`so101-demo` writes an NPZ archive readable with `numpy.load(path, allow_pickle=False)`:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer 1 |
| `costs` | Float32 array `[nx, ny, nz]`, nonnegative semantic penalties |
| `observed` | Boolean array of identical shape; distinguishes observations from unknown cells |
| `origin_m` | XYZ lower corner of the grid in the named coordinate frame |
| `voxel_size_m` | Scalar cubic cell size in metres |
| `frame` | Frame identifier; the demo uses `synthetic_base`, real data should use a calibrated robot base |
| `axis_order` | `xyz`, not image row/column order |

Cell `[i,j,k]` has centre `origin_m + voxel_size_m * ([i,j,k] + 0.5)`. To use a cost array in PyTorch, call `torch.from_numpy(array)` and move it to the desired device. Agree explicitly on interpolation and out-of-bounds behavior before using it in a differentiable planner.

An unobserved cell with zero stored cost is **unknown**, not proven free. Semantic cost is separate from occupancy and hard collision constraints. The fixture marks only sampled surfaces as observed and does not implement ray tracing, free-space carving, uncertainty, or obstacle inflation. Production records also need timestamps, camera intrinsics, depth units, calibration version and prompt/model provenance.

## First working milestone

1. **Both:** install core; run tests and synthetic demo; inspect the saved field.
2. **Perception owner:** get RGB-D capture working on Ubuntu, confirm depth scale and alignment, save a small local frame sequence with intrinsics and timestamps.
3. **Planning owner:** select and pin the SO-101 URDF; verify joint names, units, limits and FK against known poses. Develop offline using synthetic fields.
4. **Both:** estimate and validate `T_base_camera` for the fixed camera; record measured error on held-out points. `p_base = T_base_camera @ p_camera` for homogeneous column vectors.
5. **Integration:** replace synthetic points with calibrated observations and compare geometry-only planning against manual semantic penalties before adding a VLM.
6. **Hardware:** enable a separately reviewed execution path only after physical checks. This scaffold contains no motor commands.

The proposal's later stages—learned semantic fields, trajectory optimization, flow/diffusion seeds, ACT benchmarking and closed-loop evaluation—remain research implementation work. Dependency installation is not evidence that these methods work or fit the GPU.

## Collaboration rules

Keep one clone and environment per user. Exchange code through branches and pull requests, and exchange versioned data artifacts through an agreed storage location. Record data/model/calibration provenance alongside results. Coordinate one GPU-heavy experiment and one robot operator at a time; a Linux account does not reserve either resource.
