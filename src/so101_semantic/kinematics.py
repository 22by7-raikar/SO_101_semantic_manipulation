"""SO-101 follower forward kinematics on the vendored "new calibration" URDF.

LeRobot reports arm joints in degrees with 0 at the middle of each calibrated range, which is
the URDF's zero convention. Joint directions are assumed to agree (sign +1) until checked on
the real arm; record any reversed joint in ``JOINT_SIGNS``. The gripper is reported 0-100 by
LeRobot and mapped linearly onto the URDF jaw limits, which is provisional.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

URDF_PATH = Path(__file__).resolve().parents[2] / "third_party" / "so101" / "so101_new_calib.urdf"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINT_NAMES = (*ARM_JOINTS, "gripper")
JOINT_SIGNS = dict.fromkeys(ARM_JOINTS, 1.0)
TOOL_FRAME = "gripper_frame_link"


def lerobot_to_urdf(
    positions: dict[str, float], jaw_limits_rad: tuple[float, float]
) -> dict[str, float]:
    """Convert LeRobot follower readings (degrees, gripper 0-100) to URDF joint radians."""
    missing = [name for name in JOINT_NAMES if name not in positions]
    if missing:
        raise KeyError(f"Missing joints: {missing}")
    q = {name: JOINT_SIGNS[name] * np.deg2rad(positions[name]) for name in ARM_JOINTS}
    lower, upper = jaw_limits_rad
    q["gripper"] = lower + np.clip(positions["gripper"], 0.0, 100.0) / 100.0 * (upper - lower)
    return q


class SO101Model:
    """Kinematic model; meshes are not loaded."""

    def __init__(self, urdf_path: Path = URDF_PATH) -> None:
        import yourdfpy

        self.urdf = yourdfpy.URDF.load(
            str(urdf_path),
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        if set(self.urdf.actuated_joint_names) != set(JOINT_NAMES):
            raise ValueError(f"Unexpected joints in {urdf_path}: {self.urdf.actuated_joint_names}")
        self.base_frame = self.urdf.base_link

    def limits(self, joint: str) -> tuple[float, float]:
        limit = self.urdf.joint_map[joint].limit
        return float(limit.lower), float(limit.upper)

    def fk(self, q: dict[str, float], frame: str = TOOL_FRAME) -> NDArray:
        """Return T_base_frame (4x4, metres) for URDF joint radians ``q``."""
        self.urdf.update_cfg({name: float(q[name]) for name in JOINT_NAMES})
        return np.array(self.urdf.get_transform(frame, self.base_frame))

    def fk_lerobot(self, positions: dict[str, float], frame: str = TOOL_FRAME) -> NDArray:
        return self.fk(lerobot_to_urdf(positions, self.limits("gripper")), frame)
