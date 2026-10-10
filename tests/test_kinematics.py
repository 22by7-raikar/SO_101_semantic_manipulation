import numpy as np
import pytest

from so101_semantic.kinematics import JOINT_NAMES, URDF_PATH, SO101Model, lerobot_to_urdf


@pytest.fixture(scope="module")
def model():
    return SO101Model()


def zero_pose(gripper=0.0):
    return {**dict.fromkeys(JOINT_NAMES[:-1], 0.0), "gripper": gripper}


def test_vendored_urdf_present_and_joints_match(model):
    assert URDF_PATH.exists()
    assert set(model.urdf.actuated_joint_names) == set(JOINT_NAMES)


def test_zero_pose_tool_position(model):
    # Regression value for the pinned upstream URDF; not a measurement of the real arm.
    xyz = model.fk_lerobot(zero_pose())[:3, 3]
    np.testing.assert_allclose(xyz, [0.3914, 0.0, 0.2265], atol=1e-3)


def test_fk_is_rigid(model):
    t = model.fk_lerobot({**zero_pose(50.0), "shoulder_pan": 30.0, "elbow_flex": -45.0})
    r = t[:3, :3]
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_shoulder_pan_rotates_about_vertical(model):
    a = model.fk_lerobot(zero_pose())[:3, 3]
    b = model.fk_lerobot({**zero_pose(), "shoulder_pan": 90.0})[:3, 3]
    assert np.isclose(a[2], b[2], atol=1e-9)
    assert not np.allclose(a[:2], b[:2], atol=1e-3)


def test_gripper_maps_onto_jaw_limits(model):
    lower, upper = model.limits("gripper")
    assert lerobot_to_urdf(zero_pose(0.0), (lower, upper))["gripper"] == pytest.approx(lower)
    assert lerobot_to_urdf(zero_pose(100.0), (lower, upper))["gripper"] == pytest.approx(upper)
    assert lerobot_to_urdf(zero_pose(150.0), (lower, upper))["gripper"] == pytest.approx(upper)


def test_missing_joint_rejected():
    with pytest.raises(KeyError):
        lerobot_to_urdf({"shoulder_pan": 0.0}, (0.0, 1.0))
