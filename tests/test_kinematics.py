import json

import numpy as np
import pytest

from so101_semantic.kinematics import (
    ARM_JOINTS,
    JOINT_NAMES,
    SERVO_NO_LOAD_SPEED_RAD_S,
    URDF_PATH,
    JointCalibration,
    SO101Model,
    StaleJointCalibration,
    arm_to_urdf,
    calibrated_range_rad,
    clip_to_limits,
    lerobot_to_urdf,
    limit_violations,
    load_joint_calibration,
    min_segment_duration,
    urdf_to_arm,
)


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


def calibration_file(tmp_path, half_ticks):
    entries = {
        name: {
            "id": i,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 2048 - half_ticks,
            "range_max": 2048 + half_ticks,
        }
        for i, name in enumerate(JOINT_NAMES, start=1)
    }
    path = tmp_path / "follower.json"
    path.write_text(json.dumps(entries))
    return path


def test_calibrated_range_matches_lerobot_degrees():
    lower, upper = calibrated_range_rad({"range_min": 1024, "range_max": 3071})
    assert upper == pytest.approx(-lower)
    assert np.rad2deg(upper) == pytest.approx(1023.5 * 360 / 4095)


def test_joint_limits_without_calibration_use_urdf_positions_and_servo_speed(model):
    limits = model.joint_limits()
    for name in JOINT_NAMES:
        assert (limits[name].lower, limits[name].upper) == model.limits(name)
        # The URDF's velocity="10" placeholder must not leak through.
        assert limits[name].velocity == pytest.approx(SERVO_NO_LOAD_SPEED_RAD_S)
    assert SERVO_NO_LOAD_SPEED_RAD_S == pytest.approx(4.717, abs=1e-3)


def test_joint_limits_intersect_calibration(model, tmp_path):
    narrow = model.joint_limits(calibration_file(tmp_path, half_ticks=512))  # +-45 deg
    for name in JOINT_NAMES[:-1]:
        assert narrow[name].upper == pytest.approx(np.deg2rad(512 * 360 / 4095))
        assert narrow[name].lower == pytest.approx(-narrow[name].upper)
    assert (narrow["gripper"].lower, narrow["gripper"].upper) == model.limits("gripper")

    wide = model.joint_limits(calibration_file(tmp_path, half_ticks=2047))  # full turn
    for name in JOINT_NAMES:
        assert (wide[name].lower, wide[name].upper) == model.limits(name)


def test_limit_violations_and_clip(model):
    limits = model.joint_limits()
    lower, upper = model.limits("elbow_flex")
    q = {**dict.fromkeys(JOINT_NAMES, 0.0), "elbow_flex": upper + 0.1, "wrist_flex": -10.0}
    excess = limit_violations(q, limits)
    assert set(excess) == {"elbow_flex", "wrist_flex"}
    assert excess["elbow_flex"] == pytest.approx(0.1)
    assert excess["wrist_flex"] < 0
    assert limit_violations(clip_to_limits(q, limits), limits) == {}
    assert limit_violations(dict.fromkeys(JOINT_NAMES, 0.0), limits) == {}


def test_min_segment_duration_set_by_slowest_joint(model):
    limits = model.joint_limits()
    q0 = dict.fromkeys(JOINT_NAMES, 0.0)
    q1 = {**q0, "shoulder_pan": 1.0, "elbow_flex": -0.5}
    assert min_segment_duration(q0, q1, limits) == pytest.approx(1.0 / SERVO_NO_LOAD_SPEED_RAD_S)


CAL = JointCalibration(
    {
        "shoulder_pan": 0.0,
        "shoulder_lift": -3.0,
        "elbow_flex": -7.2,
        "wrist_flex": -4.0,
        "wrist_roll": -77.6,
    },
    {**dict.fromkeys(ARM_JOINTS, 1.0), "wrist_flex": -1.0},
)


def test_joint_calibration_round_trip():
    readings = {
        "shoulder_pan": 12.0,
        "shoulder_lift": -30.0,
        "elbow_flex": 80.0,
        "wrist_flex": 5.0,
        "wrist_roll": 170.0,
    }
    q = arm_to_urdf(readings, CAL)
    assert q["elbow_flex"] == pytest.approx(np.deg2rad(80.0 - 7.2))
    assert q["wrist_flex"] == pytest.approx(-np.deg2rad(5.0 - 4.0))
    back = urdf_to_arm(q, CAL)
    assert back == pytest.approx(readings)


def test_model_applies_joint_calibration(model):
    calibrated = SO101Model(joint_calibration=CAL)
    readings = {**zero_pose(), "elbow_flex": 20.0}
    expected = model.fk(lerobot_to_urdf(readings, model.limits("gripper"), CAL))
    np.testing.assert_allclose(calibrated.fk_lerobot(readings), expected)
    assert not np.allclose(calibrated.fk_lerobot(readings), model.fk_lerobot(readings))


def test_calibrated_limits_shift_with_offsets(tmp_path):
    path = calibration_file(tmp_path, half_ticks=1000)
    plain = SO101Model().joint_limits(path)["elbow_flex"]
    shifted = SO101Model(joint_calibration=CAL).joint_limits(path)["elbow_flex"]
    assert shifted.lower == pytest.approx(plain.lower - np.deg2rad(7.2))


def test_load_joint_calibration(tmp_path):
    assert load_joint_calibration(tmp_path / "missing.yaml").is_identity
    follower = tmp_path / "follower.json"
    follower.write_text("{}")
    import hashlib

    import yaml

    path = tmp_path / "joint_calibration.local.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": "fit.json",
                "lerobot_calibration_sha256": hashlib.sha256(b"{}").hexdigest(),
                "offsets_deg": CAL.offsets_deg,
                "signs": CAL.signs,
            }
        )
    )
    cal = load_joint_calibration(path, lerobot_calibration=follower)
    assert cal.offsets_deg == CAL.offsets_deg and cal.source == "fit.json"
    follower.write_text('{"recalibrated": true}')
    with pytest.raises(StaleJointCalibration):
        load_joint_calibration(path, lerobot_calibration=follower)


def test_bad_sign_rejected():
    with pytest.raises(ValueError):
        JointCalibration(signs={"elbow_flex": 2.0})
