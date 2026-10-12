import numpy as np
import pytest
from scipy.optimize import brentq

from so101_semantic.fk_fit import fit_offsets, fit_with_sign_check, tip_position, to_urdf
from so101_semantic.kinematics import JOINT_NAMES, SO101Model

TRUE_OFFSETS = {"shoulder_lift": 3.9, "elbow_flex": -8.8, "wrist_flex": -6.5, "wrist_roll": 2.0}
SHEET_YAW, SHEET_XY, TABLE_Z = np.deg2rad(-87.0), np.array([0.12, 0.14]), -0.0024


@pytest.fixture(scope="module")
def model():
    return SO101Model()


@pytest.fixture(scope="module")
def touches(model):
    """Synthetic touches: random arm poses with the shoulder solved so the true tip is on the
    table; each touch's mark is where the tip lands, expressed in the sheet frame."""
    rot = np.array(
        [[np.cos(SHEET_YAW), -np.sin(SHEET_YAW)], [np.sin(SHEET_YAW), np.cos(SHEET_YAW)]]
    )
    rng = np.random.default_rng(0)
    readings, marks = [], []
    while len(readings) < 16:
        pose = {
            "shoulder_pan": rng.uniform(-40, 40),
            "elbow_flex": rng.uniform(0, 95),
            "wrist_flex": rng.uniform(-60, 60),
            "wrist_roll": rng.uniform(-60, 60),
            "gripper": 0.0,
        }

        def height(lift, pose=pose):
            q = to_urdf({**pose, "shoulder_lift": lift}, TRUE_OFFSETS, {})
            return tip_position(model, q)[2] - TABLE_Z

        grid = np.linspace(-100, 100, 41)
        heights = [height(x) for x in grid]
        crossings = [i for i in range(40) if heights[i] * heights[i + 1] < 0]
        if not crossings:
            continue
        lift = brentq(height, grid[crossings[0]], grid[crossings[0] + 1], xtol=1e-10)
        r = {**pose, "shoulder_lift": lift}
        tip = tip_position(model, to_urdf(r, TRUE_OFFSETS, {}))
        if not 0.10 < np.hypot(tip[0] - 0.0388, tip[1]) < 0.35:
            continue
        readings.append(r)
        marks.append(rot.T @ (tip[:2] - SHEET_XY))
    return readings, np.array(marks)


def test_conversion_applies_offset_then_sign():
    q = to_urdf(dict.fromkeys(JOINT_NAMES, 10.0), {"elbow_flex": -4.0}, {"wrist_flex": -1.0})
    assert q["elbow_flex"] == pytest.approx(np.deg2rad(6.0))
    assert q["wrist_flex"] == pytest.approx(np.deg2rad(-10.0))


def test_fit_recovers_offsets_from_exact_touches(model, touches):
    readings, marks = touches
    fit = fit_offsets(model, readings, marks)
    for joint, value in TRUE_OFFSETS.items():
        assert fit["offsets_deg"][joint] == pytest.approx(value, abs=0.3), joint
    assert fit["table_z_m"] == pytest.approx(TABLE_Z, abs=5e-4)
    assert fit["rms_m"] < 5e-4


def test_fit_with_measurement_noise(model, touches):
    readings, marks = touches
    noisy = np.asarray(marks) + np.random.default_rng(1).normal(scale=0.001, size=np.shape(marks))
    fit = fit_offsets(model, readings, noisy)
    for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert fit["offsets_deg"][joint] == pytest.approx(TRUE_OFFSETS[joint], abs=1.5), joint
    assert fit["rms_m"] < 3e-3


def test_zero_offset_model_shows_the_error(model, touches):
    readings, marks = touches
    assert fit_offsets(model, readings, marks, joints=())["rms_m"] > 5e-3


def test_sign_check_prefers_true_signs(model, touches):
    readings, marks = touches
    best = fit_with_sign_check(model, readings, marks)[0]
    assert all(best["signs"][j] == 1.0 for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))
