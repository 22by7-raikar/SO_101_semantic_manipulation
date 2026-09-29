import numpy as np
import pytest

from so101_semantic.geometry import backproject, transform_points


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
