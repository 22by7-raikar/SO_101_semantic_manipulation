"""Fit joint zero offsets from touch points: the fixed-jaw tip placed on marks of a printed grid.

Each touch gives LeRobot joint readings (degrees) and the mark's position on the sheet. Unknowns
are a zero offset per fitted joint (degrees, added to the LeRobot reading), the sheet's pose on
the table (x, y, yaw in the base frame) and the table height. The shoulder_pan offset is not
fitted: it is indistinguishable from the sheet's yaw. The fit residual (tip vs. mark, mm) is the
FK check.

  uv run --no-sync python -m so101_semantic.fk_fit calibration/local/fk_touch/<file>.json
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from so101_semantic.kinematics import ARM_JOINTS, JOINT_NAMES, SO101Model

# Point of the fixed jaw in the gripper_link frame (m): the lowest mesh vertex along the jaw.
FIXED_JAW_TIP = np.array([-0.0089, -0.0002, -0.1044])
FIT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# wrist_roll turns all the way round, so its calibrated zero can be anywhere (about -75 deg on our
# follower). Its offset is bounded by +-180 deg and fitted from several starts.
ROLL_STARTS_DEG = tuple(-157.5 + 45.0 * i for i in range(8))


def to_urdf(
    readings: dict[str, float], offsets_deg: dict[str, float], signs: dict[str, float]
) -> dict[str, float]:
    """LeRobot degrees -> URDF radians with per-joint offset and sign (gripper ignored)."""
    q = {
        name: signs.get(name, 1.0) * np.deg2rad(readings[name] + offsets_deg.get(name, 0.0))
        for name in ARM_JOINTS
    }
    q["gripper"] = 0.0  # the fixed jaw does not depend on the gripper joint
    return q


def tip_position(model: SO101Model, q: dict[str, float]) -> NDArray:
    t = model.fk(q, "gripper_link")
    return t[:3, :3] @ FIXED_JAW_TIP + t[:3, 3]


def _kabsch_2d(src: NDArray, dst: NDArray) -> tuple[float, NDArray]:
    """Yaw and translation mapping 2D points src onto dst in the least-squares sense."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    h = (src - sc).T @ (dst - dc)
    yaw = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return float(yaw), dc - rot @ sc


def fit_offsets(
    model: SO101Model,
    readings: list[dict[str, float]],
    marks_m: NDArray,
    signs: dict[str, float] | None = None,
    joints: tuple[str, ...] = FIT_JOINTS,
    bound_deg: float = 30.0,
    roll_starts_deg: tuple[float, ...] = (0.0,),
) -> dict:
    """Least-squares fit of joint offsets, sheet pose and table height to touch points.

    ``marks_m`` is [n, 2]: each touched mark's position in the sheet frame (metres).
    Offsets are bounded by ``bound_deg``, except wrist_roll (+-180, from each of
    ``roll_starts_deg``; the best fit wins). Returns offsets (deg), sheet pose, table height and
    per-touch residuals (m).
    """
    from scipy.optimize import least_squares

    signs = signs or {}
    marks_m = np.asarray(marks_m, dtype=float)

    def tips(x: NDArray) -> NDArray:
        offsets = dict(zip(joints, x[: len(joints)], strict=True))
        return np.array([tip_position(model, to_urdf(r, offsets, signs)) for r in readings])

    def residuals(x: NDArray) -> NDArray:
        yaw, tx, ty, z0 = x[len(joints) :]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        marks_xy = marks_m @ rot.T + [tx, ty]
        predicted = tips(x)
        return np.concatenate([predicted[:, :2] - marks_xy, predicted[:, 2:] - z0], axis=1).ravel()

    bounds = np.array([180.0 if j == "wrist_roll" else bound_deg for j in joints])
    lower = np.concatenate([-bounds, [-np.inf] * 4])
    upper = np.concatenate([bounds, [np.inf] * 4])
    fit = None
    for roll in roll_starts_deg if "wrist_roll" in joints else (0.0,):
        offsets0 = np.array([roll if j == "wrist_roll" else 0.0 for j in joints])
        start_tips = tips(np.concatenate([offsets0, np.zeros(4)]))
        yaw0, t0 = _kabsch_2d(marks_m, start_tips[:, :2])
        x0 = np.concatenate([offsets0, [yaw0, *t0, start_tips[:, 2].mean()]])
        trial = least_squares(residuals, x0, bounds=(lower, upper), diff_step=1e-4, x_scale="jac")
        if fit is None or trial.cost < fit.cost:
            fit = trial
    per_touch = np.linalg.norm(fit.fun.reshape(-1, 3), axis=1)
    yaw, tx, ty, z0 = fit.x[len(joints) :]
    return {
        "offsets_deg": dict(zip(joints, (float(v) for v in fit.x[: len(joints)]), strict=True)),
        "signs": {j: signs.get(j, 1.0) for j in ARM_JOINTS},
        "sheet_pose": {"x_m": float(tx), "y_m": float(ty), "yaw_deg": float(np.rad2deg(yaw))},
        "table_z_m": float(z0),
        "residual_m": per_touch.tolist(),
        "rms_m": float(np.sqrt(np.mean(per_touch**2))),
        "max_m": float(per_touch.max()),
    }


def fit_with_sign_check(
    model: SO101Model, readings: list[dict[str, float]], marks_m: NDArray
) -> list[dict]:
    """Fit under every sign combination of the pitch joints; best (lowest RMS) first."""
    results = []
    for flips in itertools.product((1.0, -1.0), repeat=3):
        signs = dict(zip(("shoulder_lift", "elbow_flex", "wrist_flex"), flips, strict=True))
        results.append(fit_offsets(model, readings, marks_m, signs))
    return sorted(results, key=lambda r: r["rms_m"])


def load_touches(path: Path) -> tuple[list[dict[str, float]], NDArray]:
    record = json.loads(path.read_text())
    touches = record["touches"]
    readings = [{name: t["readings"][name] for name in JOINT_NAMES} for t in touches]
    return readings, np.array([t["mark_m"] for t in touches])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("touch_file", type=Path)
    parser.add_argument("--save", action="store_true", help="Write <touch_file>.fit.json")
    args = parser.parse_args()
    readings, marks = load_touches(args.touch_file)
    model = SO101Model()
    zero = fit_offsets(model, readings, marks, joints=())
    print(f"{len(readings)} touches")
    rms, worst = zero["rms_m"] * 1000, zero["max_m"] * 1000
    print(f"current model (no offsets): RMS {rms:.1f} mm, max {worst:.1f} mm")
    results = fit_with_sign_check(model, readings, marks)
    mirrored = fit_with_sign_check(model, readings, marks * [-1.0, 1.0])
    if mirrored[0]["rms_m"] < results[0]["rms_m"]:
        print("NOTE: marks fit better mirrored left/right; check which side is L vs R")
        results, marks = mirrored, marks * [-1.0, 1.0]
    for r in results[:3]:
        signs = " ".join(f"{j}={int(s):+d}" for j, s in r["signs"].items() if j in FIT_JOINTS[:3])
        print(f"signs {signs}: RMS {r['rms_m'] * 1000:.1f} mm, max {r['max_m'] * 1000:.1f} mm")
    # Refit the best signs with wrist_roll started all round the circle.
    best = fit_offsets(model, readings, marks, results[0]["signs"], roll_starts_deg=ROLL_STARTS_DEG)
    print(
        f"refit, wrist_roll from {len(ROLL_STARTS_DEG)} starts: RMS {best['rms_m'] * 1000:.1f} mm"
    )
    print("best fit offsets (deg, added to LeRobot readings):")
    for j, v in best["offsets_deg"].items():
        print(f"  {j:14s} {v:+6.1f}")
    print(f"table z {best['table_z_m'] * 1000:.1f} mm; sheet {best['sheet_pose']}")
    print("per-touch residual (mm):", " ".join(f"{e * 1000:.1f}" for e in best["residual_m"]))
    if args.save:
        out = args.touch_file.with_suffix(".fit.json")
        record = json.loads(args.touch_file.read_text())
        best = {**best, "touch_file": str(args.touch_file), "paper": record.get("paper")}
        best["sheet_scale"] = record.get("sheet_scale", [1.0, 1.0])
        best["calibration_sha256"] = record.get("calibration_sha256")
        best["zero_offset_rms_m"] = zero["rms_m"]
        out.write_text(json.dumps(best, indent=1))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
