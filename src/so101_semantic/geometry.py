"""Metric camera geometry. Depth must already be aligned to the chosen intrinsics."""

import numpy as np
from numpy.typing import NDArray


def backproject(depth_m: NDArray, intrinsics: NDArray) -> NDArray:
    """Return valid XYZ camera points (metres); drop zero, negative and nonfinite depth."""
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Expected HxW depth and finite 3x3 intrinsics")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Focal lengths must be positive")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("Expected rectified pinhole intrinsics without skew")
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    return np.column_stack(
        ((u[valid] - k[0, 2]) * z / k[0, 0], (v[valid] - k[1, 2]) * z / k[1, 1], z)
    )


def transform_points(points: NDArray, transform: NDArray) -> NDArray:
    """Apply T_target_source to row-wise XYZ points, without changing metric units."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 points")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 rigid transform")
    r = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
    ):
        raise ValueError("Transform must be rigid and right-handed")
    # Fixed-width XYZ contraction avoids platform BLAS behavior for tall, narrow arrays.
    return np.einsum("nj,ij->ni", points, r) + transform[:3, 3]
