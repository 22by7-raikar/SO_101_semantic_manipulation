"""SO-101 follower forward kinematics on the vendored "new calibration" URDF.

LeRobot reports arm joints in degrees with 0 at the middle of each calibrated range, which is
the URDF's zero convention. Joint directions are assumed to agree (sign +1) until checked on
the real arm; record any reversed joint in ``JOINT_SIGNS``. The gripper is reported 0-100 by
LeRobot and mapped linearly onto the URDF jaw limits, which is provisional.

Velocity and effort come from the servo datasheet, not the URDF, whose ``velocity="10"
effort="10"`` on every joint are upstream placeholders.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

URDF_PATH = Path(__file__).resolve().parents[2] / "third_party" / "so101" / "so101_new_calib.urdf"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINT_NAMES = (*ARM_JOINTS, "gripper")
JOINT_SIGNS = dict.fromkeys(ARM_JOINTS, 1.0)
TOOL_FRAME = "gripper_frame_link"

# Feetech STS3215-C018 (12 V) datasheet: 0.222 s/60 deg no load, 30 kg*cm stall. These are
# physical ceilings, not safe planning values. Our follower servos read 11.8-11.9 V with a 14.0 V
# Max_Voltage_Limit (7.4 V variants ship with 8.0 V), consistent with C018; labels not checked.
STS3215_TICKS = 4096
SERVO_NO_LOAD_SPEED_RAD_S = float(np.deg2rad(60.0) / 0.222)
SERVO_STALL_TORQUE_NM = 30.0 * 0.0980665


@dataclass(frozen=True)
class JointLimit:
    lower: float  # rad
    upper: float  # rad
    velocity: float  # rad/s
    effort: float  # N*m


def calibrated_range_rad(entry: dict) -> tuple[float, float]:
    """Joint range in URDF radians from one LeRobot calibration entry (degrees mode).

    LeRobot reports ``(raw - mid) * 360 / (ticks - 1)`` with ``mid`` the range centre, so the
    range is symmetric about zero and independent of drive direction.
    """
    half = (entry["range_max"] - entry["range_min"]) / 2 * 2 * np.pi / (STS3215_TICKS - 1)
    return -half, half


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

    def joint_limits(self, calibration_path: Path | None = None) -> dict[str, JointLimit]:
        """Per-joint limits; arm positions are the intersection of URDF and calibrated ranges.

        The gripper keeps the URDF range because LeRobot maps its calibrated range onto it.
        """
        calibration = json.loads(calibration_path.read_text()) if calibration_path else {}
        limits = {}
        for name in JOINT_NAMES:
            lower, upper = self.limits(name)
            if name in ARM_JOINTS and name in calibration:
                cal_lower, cal_upper = calibrated_range_rad(calibration[name])
                lower, upper = max(lower, cal_lower), min(upper, cal_upper)
            limits[name] = JointLimit(
                lower, upper, SERVO_NO_LOAD_SPEED_RAD_S, SERVO_STALL_TORQUE_NM
            )
        return limits

    def fk(self, q: dict[str, float], frame: str = TOOL_FRAME) -> NDArray:
        """Return T_base_frame (4x4, metres) for URDF joint radians ``q``."""
        self.urdf.update_cfg({name: float(q[name]) for name in JOINT_NAMES})
        return np.array(self.urdf.get_transform(frame, self.base_frame))

    def fk_lerobot(self, positions: dict[str, float], frame: str = TOOL_FRAME) -> NDArray:
        return self.fk(lerobot_to_urdf(positions, self.limits("gripper")), frame)

    def collision_spheres(
        self, q: dict[str, float], spheres: dict[str, tuple[NDArray, NDArray]] | None = None
    ) -> tuple[NDArray, NDArray, list[str]]:
        """Sphere centres [n, 3] in the base frame, radii [n] and each sphere's link, at ``q``."""
        from so101_semantic.collision import load_spheres

        spheres = load_spheres() if spheres is None else spheres
        self.urdf.update_cfg({name: float(q[name]) for name in JOINT_NAMES})
        centres, radii, links = [], [], []
        for link, (local, r) in spheres.items():
            t = np.array(self.urdf.get_transform(link, self.base_frame))
            centres.append(local @ t[:3, :3].T + t[:3, 3])
            radii.append(r)
            links += [link] * len(r)
        return np.vstack(centres), np.concatenate(radii), links

    def self_collision_distances(self, q: dict[str, float]) -> dict[tuple[str, str], float]:
        """Signed sphere distance (m) for each checked link pair; negative means overlapping.

        Pairs that are adjacent or proven unable to touch are skipped (see the spheres config).
        Spheres overestimate the links, so this is conservative: a negative value can be a
        near miss, but a real contact always gives a negative value.
        """
        from so101_semantic.collision import load_ignored_pairs, load_spheres, self_collision_pairs

        if not hasattr(self, "_self_collision"):
            spheres = load_spheres()
            self._self_collision = (spheres, self_collision_pairs(spheres, load_ignored_pairs()))
        spheres, pairs = self._self_collision
        centres, radii, links = self.collision_spheres(q, spheres)
        links = np.array(links)
        distances = {}
        for a, b in pairs:
            ia, ib = links == a, links == b
            gap = np.linalg.norm(centres[ia][:, None] - centres[ib][None], axis=2)
            distances[(a, b)] = float((gap - radii[ia][:, None] - radii[ib][None]).min())
        return distances

    def self_collisions(self, q: dict[str, float], margin: float = 0.0) -> list[tuple[str, str]]:
        """Checked link pairs whose spheres are closer than ``margin`` (m) at ``q``."""
        return [pair for pair, d in self.self_collision_distances(q).items() if d < margin]


def limit_violations(q: dict[str, float], limits: dict[str, JointLimit]) -> dict[str, float]:
    """Joints outside their position limits, mapped to the signed excess in radians."""
    excess = {}
    for name, limit in limits.items():
        if q[name] < limit.lower:
            excess[name] = q[name] - limit.lower
        elif q[name] > limit.upper:
            excess[name] = q[name] - limit.upper
    return excess


def clip_to_limits(q: dict[str, float], limits: dict[str, JointLimit]) -> dict[str, float]:
    return {name: float(np.clip(q[name], lim.lower, lim.upper)) for name, lim in limits.items()}


def min_segment_duration(
    q0: dict[str, float], q1: dict[str, float], limits: dict[str, JointLimit]
) -> float:
    """Shortest time (s) for a straight joint-space move that respects every velocity limit."""
    return max(abs(q1[name] - q0[name]) / lim.velocity for name, lim in limits.items())
