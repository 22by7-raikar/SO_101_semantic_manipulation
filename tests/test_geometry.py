import numpy as np
import pytest

from so101_semantic.geometry import (
    backproject,
    fit_plane,
    plane_height,
    remove_table,
    transform_plane,
    transform_points,
)


def test_backproject_metric_depth_and_invalid_pixels():
    depth = np.array([[1.0, 2.0, 0.0], [np.nan, -1.0, np.inf]])
    k = np.diag([2.0, 2.0, 1.0])
    np.testing.assert_allclose(backproject(depth, k), [[0, 0, 1], [1, 0, 2]])


def test_camera_to_base_rotation_translation():
    transform = np.array([[0, -1, 0, 0.2], [1, 0, 0, 0.3], [0, 0, 1, 0.4], [0, 0, 0, 1]])
    np.testing.assert_allclose(transform_points([[1, 0, 2]], transform), [[0.2, 1.3, 2.4]])


def test_empty_depth():
    assert backproject(np.zeros((2, 2)), np.eye(3)).shape == (0, 3)


@pytest.mark.parametrize("focal", [0, -1, np.nan])
def test_bad_intrinsics(focal):
    with pytest.raises(ValueError):
        backproject(np.ones((1, 1)), np.diag([focal, 1, 1]))


def test_reflection_is_not_rigid_rotation():
    with pytest.raises(ValueError):
        transform_points([[0, 0, 0]], np.diag([-1, 1, 1, 1]))


def test_bulk_identity_transform_is_finite():
    points = np.linspace(-0.1, 0.5, 3072).reshape(-1, 3)
    with np.errstate(all="raise"):
        result = transform_points(points, np.eye(4))
    np.testing.assert_allclose(result, points)


TABLE_Z = -0.0024  # base mesh bottom: the arm base, camera stand and table share this surface
BLOCK = (np.array([0.20, -0.02, TABLE_Z]), np.array([0.24, 0.02, TABLE_Z + 0.04]))  # 4 cm block
K = np.array([[428.9, 0, 422.4], [0, 428.9, 244.7], [0, 0, 1.0]])  # D435 depth, 848x480


def camera_on_stand(position=(0.55, 0.35, 0.55), target=(0.2, 0.0, 0.0)):
    """T_base_camera for an optical frame (x right, y down, z forward) looking at ``target``."""
    z = np.subtract(target, position) / np.linalg.norm(np.subtract(target, position))
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    t[:3, 3] = position
    return t


def render_depth(t_base_camera, noise=0.001, seed=0):
    """Depth image (m) of the table and the block, by ray casting; 0 where nothing is hit."""
    v, u = np.indices((480, 848))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)
    origin, dirs = t_base_camera[:3, 3], rays @ t_base_camera[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(dirs[..., 2] < 0, (TABLE_Z - origin[2]) / dirs[..., 2], np.inf)
        t0, t1 = (BLOCK[0] - origin) / dirs, (BLOCK[1] - origin) / dirs
        near = np.minimum(t0, t1).max(-1)
        far = np.maximum(t0, t1).min(-1)
    hit = (near <= far) & (near > 0)
    depth = np.where(hit, np.minimum(depth, near), depth)  # rays have z = 1: s is depth
    depth = depth + np.random.default_rng(seed).normal(scale=noise, size=depth.shape)
    return np.where(np.isfinite(depth) & (depth < 3), depth, 0.0)


def test_table_plane_from_depth_matches_the_base_frame_table():
    t_base_camera = camera_on_stand()
    points = backproject(render_depth(t_base_camera), K)
    plane, inliers = fit_plane(points)
    assert inliers.mean() > 0.9
    assert plane[3] > 0  # normal toward the camera: up
    in_base = transform_plane(plane, t_base_camera)
    # The consistency check once T_base_camera is calibrated: flat, at the base mesh bottom.
    assert np.degrees(np.arccos(in_base[2])) < 0.5
    assert -in_base[3] == pytest.approx(TABLE_Z, abs=0.002)


def test_remove_table_keeps_the_block():
    t_base_camera = camera_on_stand()
    points = backproject(render_depth(t_base_camera), K)
    plane, _ = fit_plane(points)
    kept = transform_points(remove_table(points, plane, min_height=0.01), t_base_camera)
    assert len(kept) > 200
    assert np.all(kept[:, 2] > TABLE_Z + 0.008)
    assert np.all((kept[:, :2] > BLOCK[0][:2] - 0.01) & (kept[:, :2] < BLOCK[1][:2] + 0.01))
    assert kept[:, 2].max() == pytest.approx(BLOCK[1][2], abs=0.005)


def test_normal_hint_rejects_a_larger_wall():
    rng = np.random.default_rng(1)
    table = np.column_stack([rng.uniform(0, 0.5, (3000, 2)), np.zeros(3000)])
    wall = np.column_stack(
        [np.full(6000, 0.6), rng.uniform(-0.5, 0.5, 6000), rng.uniform(0, 0.8, 6000)]
    )
    points = np.vstack([table, wall])
    assert abs(fit_plane(points)[0][0]) > 0.99  # the wall wins without a hint
    plane, inliers = fit_plane(points, normal_hint=[0, 0, 1])
    np.testing.assert_allclose(plane, [0, 0, 1, 0], atol=1e-3)
    assert inliers[:3000].all() and inliers[3000:].mean() < 0.01  # wall foot within 5 mm
    assert plane_height([[0.1, 0.1, 0.05]], plane) == pytest.approx([0.05], abs=1e-4)
