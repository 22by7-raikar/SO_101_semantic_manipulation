"""SO-101 follower forward kinematics on the vendored "new calibration" URDF.

LeRobot reports arm joints in degrees with 0 at the middle of each calibrated range. The URDF's
zero differs per joint, so the conversion is ``q = sign * radians(reading + offset_deg)``, with
offsets and signs in a ``JointCalibration``. Code for the real arm loads it from
``configs/joint_calibration.local.yaml`` (``SO101Model.from_config()``), written by
``fk_fit --apply``; ``SO101Model()`` uses the identity (offsets 0, signs +1), as in tests. The
gripper is reported 0-100 by LeRobot and mapped linearly onto the URDF jaw limits, which is
provisional.

Velocity and effort come from the servo datasheet, not the URDF, whose ``velocity="10"
effort="10"`` on every joint are upstream placeholders.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = ROOT / "third_party" / "so101" / "so101_new_calib.urdf"
JOINT_CALIBRATION_PATH = ROOT / "configs" / "joint_calibration.local.yaml"
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


class StaleJointCalibration(RuntimeError):
    """The joint calibration was fitted against a different LeRobot calibration file."""


@dataclass(frozen=True)
class JointCalibration:
    """Per-joint LeRobot -> URDF mapping: ``q = sign * radians(reading + offset_deg)``."""

    offsets_deg: dict[str, float] = field(default_factory=lambda: dict.fromkeys(ARM_JOINTS, 0.0))
    signs: dict[str, float] = field(default_factory=lambda: dict(JOINT_SIGNS))
    lerobot_calibration_sha256: str | None = None  # follower.json the offsets belong to
    source: str | None = None  # where the values came from (fit file), for provenance

    def __post_init__(self) -> None:
        unknown = set(self.offsets_deg) - set(ARM_JOINTS) | set(self.signs) - set(ARM_JOINTS)
        if unknown or any(abs(v) != 1.0 for v in self.signs.values()):
            raise ValueError(f"Bad joint calibration: unknown {sorted(unknown)} or sign not +-1")

    def offset(self, joint: str) -> float:
        return float(self.offsets_deg.get(joint, 0.0))

    def sign(self, joint: str) -> float:
        return float(self.signs.get(joint, 1.0))

    @property
    def is_identity(self) -> bool:
        return all(self.offset(j) == 0.0 and self.sign(j) == 1.0 for j in ARM_JOINTS)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_joint_calibration(
    path: Path = JOINT_CALIBRATION_PATH, lerobot_calibration: Path | None = None
) -> JointCalibration:
    """Joint calibration from YAML; identity if the file does not exist.

    With ``lerobot_calibration`` (the follower.json in use), raise ``StaleJointCalibration`` if
    the offsets were fitted against a different file: re-running ``lerobot-calibrate`` moves
    every zero, so old offsets would silently be wrong.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        return JointCalibration()
    record = yaml.safe_load(path.read_text()) or {}
    cal = JointCalibration(
        offsets_deg={k: float(v) for k, v in (record.get("offsets_deg") or {}).items()},
        signs={k: float(v) for k, v in (record.get("signs") or {}).items()},
        lerobot_calibration_sha256=record.get("lerobot_calibration_sha256"),
        source=record.get("source"),
    )
    if lerobot_calibration is not None and cal.lerobot_calibration_sha256:
        current = file_sha256(lerobot_calibration)
        if current != cal.lerobot_calibration_sha256:
            raise StaleJointCalibration(
                f"{path} was fitted for another LeRobot calibration than {lerobot_calibration}; "
                "redo the touch session (docs/CALIBRATION_STEPS.md) or point to the right file"
            )
    return cal


def calibrated_range_rad(
    entry: dict, joint: str | None = None, cal: JointCalibration | None = None
) -> tuple[float, float]:
    """Joint range in URDF radians from one LeRobot calibration entry (degrees mode).

    LeRobot reports ``(raw - mid) * 360 / (ticks - 1)`` with ``mid`` the range centre, so the
    reading range is symmetric about zero; the joint calibration then shifts (and may flip) it.
    """
    half = (entry["range_max"] - entry["range_min"]) / 2 * 360.0 / (STS3215_TICKS - 1)
    if joint is None or cal is None:
        return float(np.deg2rad(-half)), float(np.deg2rad(half))
    ends = [cal.sign(joint) * np.deg2rad(r + cal.offset(joint)) for r in (-half, half)]
    return float(min(ends)), float(max(ends))


def arm_to_urdf(positions: dict[str, float], cal: JointCalibration | None = None) -> dict:
    """Arm joints only: LeRobot degrees -> URDF radians."""
    cal = cal or JointCalibration()
    missing = [name for name in ARM_JOINTS if name not in positions]
    if missing:
        raise KeyError(f"Missing joints: {missing}")
    return {j: cal.sign(j) * np.deg2rad(positions[j] + cal.offset(j)) for j in ARM_JOINTS}


def urdf_to_arm(q: dict[str, float], cal: JointCalibration | None = None) -> dict[str, float]:
    """Inverse of ``arm_to_urdf``: URDF radians -> LeRobot degrees (for commands)."""
    cal = cal or JointCalibration()
    return {j: float(np.rad2deg(q[j]) * cal.sign(j) - cal.offset(j)) for j in ARM_JOINTS}


def lerobot_to_urdf(
    positions: dict[str, float],
    jaw_limits_rad: tuple[float, float],
    cal: JointCalibration | None = None,
) -> dict[str, float]:
    """Convert LeRobot follower readings (degrees, gripper 0-100) to URDF joint radians."""
    missing = [name for name in JOINT_NAMES if name not in positions]
    if missing:
        raise KeyError(f"Missing joints: {missing}")
    q = arm_to_urdf(positions, cal)
    lower, upper = jaw_limits_rad
    q["gripper"] = lower + np.clip(positions["gripper"], 0.0, 100.0) / 100.0 * (upper - lower)
    return q


class SO101Model:
    """Kinematic model; meshes are not loaded."""

    def __init__(
        self, urdf_path: Path = URDF_PATH, joint_calibration: JointCalibration | None = None
    ) -> None:
        import yourdfpy

        self.joint_calibration = joint_calibration or JointCalibration()
        self.urdf = yourdfpy.URDF.load(
            str(urdf_path),
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        if set(self.urdf.actuated_joint_names) != set(JOINT_NAMES):
            raise ValueError(f"Unexpected joints in {urdf_path}: {self.urdf.actuated_joint_names}")
        self.base_frame = self.urdf.base_link

    @classmethod
    def from_config(
        cls,
        joint_calibration_path: Path = JOINT_CALIBRATION_PATH,
        lerobot_calibration: Path | None = None,
    ) -> "SO101Model":
        """Model of our follower: joint calibration from ``configs/joint_calibration.local.yaml``,
        checked against ``lerobot_calibration`` (the follower.json in use) when given."""
        return cls(
            joint_calibration=load_joint_calibration(joint_calibration_path, lerobot_calibration)
        )

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
                cal_lower, cal_upper = calibrated_range_rad(
                    calibration[name], name, self.joint_calibration
                )
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
        return self.fk(
            lerobot_to_urdf(positions, self.limits("gripper"), self.joint_calibration), frame
        )

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
