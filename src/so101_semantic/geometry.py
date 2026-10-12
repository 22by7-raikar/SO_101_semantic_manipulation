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


def fit_plane(
    points: NDArray,
    threshold: float = 0.005,
    iterations: int = 300,
    normal_hint: NDArray | None = None,
    max_angle_deg: float = 25.0,
    max_samples: int = 20_000,
    seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Dominant plane ``[a, b, c, d]`` (unit normal, ``n . p + d = 0``) and its inlier mask.

    RANSAC on a subsample, then least squares on all inliers. ``normal_hint`` keeps only planes
    whose normal is within ``max_angle_deg`` of it (e.g. "up" once T_base_camera is known), so a
    wall or a large box top cannot win. The normal points along the hint, or without one toward
    the frame origin (the camera, for camera-frame points), i.e. up for a table seen from above.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("Expected Nx3 points, N >= 3")
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), max_samples), replace=False)]
    tri = sample[rng.integers(0, len(sample), (iterations, 3))]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > 1e-12
    normals = normals[keep] / norms[keep, None]
    anchors = tri[keep, 0]
    if normal_hint is not None:
        hint = np.asarray(normal_hint, float) / np.linalg.norm(normal_hint)
        ok = np.abs(normals @ hint) >= np.cos(np.deg2rad(max_angle_deg))
        normals, anchors = normals[ok], anchors[ok]
    if not len(normals):
        raise ValueError("No plane candidates (degenerate points or none near the normal hint)")
    offsets = -np.einsum("ni,ni->n", normals, anchors)
    counts = (np.abs(sample @ normals.T + offsets) < threshold).sum(0)
    best = int(np.argmax(counts))
    inliers = np.abs(points @ normals[best] + offsets[best]) < threshold
    centroid = points[inliers].mean(0)
    normal = np.linalg.svd(points[inliers] - centroid, full_matrices=False)[2][2]
    if normal_hint is not None:
        normal = normal if normal @ hint >= 0 else -normal
    elif -normal @ centroid < 0:  # origin on the negative side: flip
        normal = -normal
    plane = np.append(normal, -normal @ centroid)
    return plane, np.abs(points @ normal + plane[3]) < threshold


def plane_height(points: NDArray, plane: NDArray) -> NDArray:
    """Signed distance of each point above the plane (along its normal), metres."""
    return np.asarray(points, float) @ plane[:3] + plane[3]


def transform_plane(plane: NDArray, transform: NDArray) -> NDArray:
    """Express a plane given in the source frame in the target frame of T_target_source."""
    r, t = np.asarray(transform, float)[:3, :3], np.asarray(transform, float)[:3, 3]
    normal = r @ plane[:3]
    return np.append(normal, plane[3] - normal @ t)


def remove_table(points: NDArray, plane: NDArray, min_height: float = 0.01) -> NDArray:
    """Drop the table and anything below it: keep points more than ``min_height`` above the
    plane. Objects flatter than ``min_height`` (paper, tape) go with the table."""
    points = np.asarray(points, float)
    return points[plane_height(points, plane) > min_height]
