import numpy as np
import pytest

from so101_semantic.collision import load_spheres
from so101_semantic.distance_field import (
    DistanceField,
    clearance,
    clearance_jacobian,
    sphere_clearances,
)
from so101_semantic.kinematics import JOINT_NAMES, SO101Model

BOX = ([-0.1, -0.4, -0.05], [0.5, 0.4, 0.4])  # workspace box, base frame, metres
# A 9.4 cm cube off the voxel boundaries, like a block on the table in front of the arm.
CUBE_LO = np.array([0.203, -0.047, 0.0])
CUBE_HI = np.array([0.297, 0.047, 0.097])


def cube_surface(n=40_000, top_only=False, seed=0):
    """Points on the cube's faces, as depth would see them (or only its top, seen from above)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(CUBE_LO, CUBE_HI, (n, 3))
    if top_only:
        p[:, 2] = CUBE_HI[2]
        return p
    axis, side = rng.integers(0, 3, n), rng.integers(0, 2, n)
    p[np.arange(n), axis] = np.where(side, CUBE_HI[axis], CUBE_LO[axis])
    return p


def true_cube_distance(points):
    outside = np.maximum(np.maximum(CUBE_LO - points, points - CUBE_HI), 0)
    return np.linalg.norm(outside, axis=1)


@pytest.fixture(scope="module")
def field():
    return DistanceField.from_points(cube_surface(), *BOX, voxel_size_m=0.01)


def test_box_at_known_points_is_close_and_conservative(field):
    queries = []
    for d in (0.01, 0.015, 0.02, 0.05):
        queries += [
            [CUBE_HI[0] + d, 0.0, 0.05],  # in front of a face
            [0.25, CUBE_LO[1] - d, 0.05],
            [0.25, 0.0, CUBE_HI[2] + d],  # above the top
            [CUBE_HI[0] + d, CUBE_HI[1] + d, 0.05],  # off a vertical edge
        ]
    queries = np.array(queries)
    value, grad = field.query(queries)
    error = value - true_cube_distance(queries)
    assert error.max() <= 1e-6  # never further than the true surface
    assert error.min() >= -0.01  # at most one voxel closer
    # Gradient points away from the cube: unit length off faces, near unit off edges.
    norms = np.linalg.norm(grad, axis=1).reshape(-1, 4)
    assert np.allclose(norms[:, :3], 1.0, atol=0.01)
    assert np.allclose(norms[:, 3], 1.0, atol=0.15)
    assert grad[0] == pytest.approx([1, 0, 0], abs=0.05)
    assert grad[2] == pytest.approx([0, 0, 1], abs=0.05)


def test_gradient_matches_finite_differences(field):
    rng = np.random.default_rng(1)
    points = rng.uniform(*BOX, (200, 3))
    _, grad = field.query(points)
    eps = 1e-6
    for a in range(3):
        step = np.zeros(3)
        step[a] = eps
        fd = (field.query(points + step)[0] - field.query(points - step)[0]) / (2 * eps)
        np.testing.assert_allclose(grad[:, a], fd, atol=1e-4)


def test_signed_inside_with_filled_occupancy():
    top = DistanceField.from_points(cube_surface(top_only=True), *BOX, fill_down=True)
    beside = np.array([[CUBE_HI[0] + 0.02, 0.0, 0.03]])  # level with the cube's lower half
    assert top.query(beside)[0][0] == pytest.approx(0.02, abs=0.01)
    assert top.query([[0.25, 0.0, 0.05]])[0][0] < -0.03  # cube centre, inside
    # Without filling, the unseen sides read as free: the top is the only obstacle.
    unfilled = DistanceField.from_points(cube_surface(top_only=True), *BOX)
    assert unfilled.query(beside)[0][0] > 0.05


def test_points_outside_the_box_are_ignored_and_queries_clamp():
    inside = np.array([[0.2, 0.0, 0.1]])
    field = DistanceField.from_points(
        np.vstack([inside, [[2.0, 0, 0], [0, 0, -1.0], [np.nan, 0, 0]]]), *BOX
    )
    assert field.occupied.sum() == 1
    # Beyond the +x face the value clamps to the boundary and the x gradient is zero.
    value, grad = field.query([[0.6, 0.0, 0.1], [0.495, 0.0, 0.1]])
    assert value[0] == pytest.approx(value[1], abs=1e-6)
    assert grad[0, 0] == 0.0


def test_empty_cloud_is_far_everywhere():
    field = DistanceField.from_points(np.empty((0, 3)), *BOX)
    value, grad = field.query(np.random.default_rng(2).uniform(*BOX, (50, 3)))
    assert np.all(value > 0.5) and np.all(np.isfinite(value))
    assert np.abs(grad).max() < 1e-9


def test_min_points_rejects_speckle():
    lone = np.array([[0.1, 0.1, 0.1]])
    assert not DistanceField.from_points(lone, *BOX, min_points=2).occupied.any()


def test_rejects_bad_boxes():
    with pytest.raises(ValueError):
        DistanceField.from_points(np.zeros((1, 3)), [0, 0, 0], [0, 0.1, 0.1])


def test_torch_query_matches_numpy(field):
    torch = pytest.importorskip("torch")
    points = np.random.default_rng(3).uniform(
        np.array(BOX[0]) - 0.05, np.array(BOX[1]) + 0.05, (500, 3)
    )
    x = torch.tensor(points, requires_grad=True)
    y = field.torch()(x)
    y.sum().backward()
    value, grad = field.query(points)
    np.testing.assert_allclose(y.detach().numpy(), value, atol=1e-6)
    np.testing.assert_allclose(x.grad.numpy(), grad, atol=1e-4)
    assert field.torch()(x.reshape(10, 50, 3)).shape == (10, 50)


@pytest.fixture(scope="module")
def model():
    return SO101Model()


def reach_pose():
    q = dict.fromkeys(JOINT_NAMES, 0.0)
    q["shoulder_lift"], q["elbow_flex"] = 0.3, -0.4
    return q


def test_clearance_matches_brute_force(model):
    q = reach_pose()
    spheres = {k: v for k, v in load_spheres().items() if k != "base_link"}
    centres, radii, _ = model.collision_spheres(q, spheres)
    # Obstacle points 3 cm from the gripper's lowest sphere surface.
    low = int(np.argmin(centres[:, 2] - radii))
    cloud = (
        centres[low]
        - [0, 0, radii[low] + 0.03]
        + np.random.default_rng(4).normal(scale=0.002, size=(200, 3))
    )
    field = DistanceField.from_points(cloud, *BOX)
    exact = np.min(np.linalg.norm(centres[:, None] - cloud[None], axis=2).min(1) - radii)
    got = clearance(model, q, field)
    assert exact - 0.015 <= got <= exact + 1e-6


def test_base_link_is_excluded_from_clearance(model):
    q = reach_pose()
    centres, radii, _ = model.collision_spheres(q, {"base_link": load_spheres()["base_link"]})
    low = int(np.argmin(centres[:, 2] - radii))  # below the table: masking only
    cloud = (
        centres[low]
        - [0, 0, radii[low] - 0.005]
        + np.random.default_rng(5).normal(scale=0.002, size=(100, 3))
    )
    field = DistanceField.from_points(cloud, *BOX)
    _, links = sphere_clearances(model, q, field)
    assert "base_link" not in links
    with_base, _ = sphere_clearances(model, q, field, exclude=())
    assert with_base.min() < 0 < clearance(model, q, field)


def test_clearance_jacobian_matches_finite_differences(model):
    q = reach_pose()
    field = DistanceField.from_points(cube_surface(), *BOX)
    values, jac, links = clearance_jacobian(model, q, field)
    assert jac.shape == (len(links), len(JOINT_NAMES))
    for j, name in enumerate(JOINT_NAMES):
        plus, minus = dict(q), dict(q)
        plus[name] += 1e-5
        minus[name] -= 1e-5
        fd = sphere_clearances(model, plus, field)[0] - sphere_clearances(model, minus, field)[0]
        np.testing.assert_allclose(jac[:, j], fd / 2e-5, atol=1e-3)
    np.testing.assert_allclose(values, sphere_clearances(model, q, field)[0])
