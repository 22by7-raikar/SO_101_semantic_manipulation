"""Record touch points for the joint-offset fit: fixed-jaw tip on marks of a sheet on the table.

Default sheet: the printed calibration sheet (scripts/make_calib_sheet.py; --paper calib-letter,
the default, or calib-a4: name the layout you printed), taped flat in front of the arm with its
N row toward the base. Its nine crosshairs are labelled NL..FR. Fallback: a plain sheet folded in
half both ways (--paper a4/letter), long edge across the
front, near edge ~3-5 cm from the base; marks are corners, edge midpoints and centre, named
looking from behind the robot in the direction it faces:

  far row:    FL  FC  FR
  middle row: ML  MC  MR
  near row:   NL  NC  NR     (robot base below this row)

With the follower's torque off (teleop stopped), close the gripper, hold the arm so the point of
the fixed jaw rests on a mark, type the mark's name and press ENTER. Touch every reachable mark at
least once, and several again with the wrist rolled or the arm bent differently: 12-18 touches.
Read-only: never writes to the motors.

  uv run --no-sync python scripts/record_touch_points.py
  uv run --no-sync python -m so101_semantic.fk_fit calibration/local/fk_touch/<file>.json --save
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from so101_semantic import calib_sheet
from so101_semantic.kinematics import JOINT_NAMES
from so101_semantic.pose import default_calibration, default_port, open_bus

ROOT = Path(__file__).resolve().parents[1]
PAPER_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}  # long edge, short edge


def marks_m(paper: str) -> dict[str, tuple[float, float]]:
    """Mark positions in the sheet frame: x along the near edge from left, y away from the arm."""
    if paper.startswith("calib-"):
        return calib_sheet.marks_m(paper.removeprefix("calib-"))
    length, width = (v / 1000 for v in PAPER_MM[paper])
    xs = {"L": 0.0, "C": length / 2, "R": length}
    ys = {"N": 0.0, "M": width / 2, "F": width}
    return {row + col: (xs[col], ys[row]) for row in "NMF" for col in "LCR"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    calib = [f"calib-{name}" for name in sorted(calib_sheet.LAYOUTS)]
    parser.add_argument("--paper", choices=[*calib, *sorted(PAPER_MM)], default="calib-letter")
    parser.add_argument("--port", default=default_port())
    parser.add_argument("--calibration", type=Path, default=default_calibration())
    args = parser.parse_args()
    if not args.port:
        raise SystemExit("No follower port; pass --port or set robot.follower_port")

    marks = marks_m(args.paper)
    out_dir = ROOT / "calibration" / "local" / "fk_touch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"touch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    record = {
        "paper": args.paper,
        "git_sha": sha,
        "calibration_file": str(args.calibration),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "touches": [],
    }

    bus, calibration = open_bus(args.port, args.calibration)
    try:
        stored = bus.read_calibration()
        stale = [n for n in JOINT_NAMES if stored[n] != calibration[n]]
        if stale:
            raise SystemExit(f"Motor calibration differs from {args.calibration} for {stale}")
        if any(bus.read("Torque_Enable", n) for n in JOINT_NAMES):
            raise SystemExit("Torque is on; stop teleop so the arm can be moved by hand")
        print(f"Marks: {' '.join(marks)}. Type a mark name + ENTER to record, 'u' undo, 'q' quit.")
        while True:
            entry = input(f"[{len(record['touches'])} recorded] mark> ").strip().upper()
            if entry == "Q":
                break
            if entry == "U":
                if record["touches"]:
                    print(f"removed {record['touches'].pop()['label']}")
                continue
            if entry not in marks:
                print(f"unknown mark {entry!r}")
                continue
            positions = bus.sync_read("Present_Position")
            record["touches"].append(
                {
                    "label": entry,
                    "mark_m": list(marks[entry]),
                    "readings": {n: float(positions[n]) for n in JOINT_NAMES},
                    "time": time.time(),
                }
            )
            out.write_text(json.dumps(record, indent=1))  # saved after every touch
            joints = " ".join(f"{n}={positions[n]:.1f}" for n in JOINT_NAMES)
            print(f"  {entry}: {joints}")
    finally:
        bus.disconnect(disable_torque=False)
    if record["touches"]:
        print(f"{len(record['touches'])} touches saved to {out}")
    else:
        print("No touches recorded; nothing saved")


if __name__ == "__main__":
    main()
