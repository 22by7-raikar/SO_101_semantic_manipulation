"""Read-only SO-101 follower pose: LeRobot joint readings and model tool position.

Opens the follower motor bus, reads Present_Position and calibration registers, and closes the
port without writing (torque state is left unchanged). Do not run during teleoperation or
calibration, which use the same serial port.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from so101_semantic.kinematics import JOINT_NAMES, SO101Model

ROOT = Path(__file__).resolve().parents[2]


def default_calibration() -> Path:
    base = Path(os.environ.get("HF_LEROBOT_CALIBRATION", ROOT / "calibration/local/lerobot"))
    return base / "robots" / "so_follower" / "follower.json"


def default_port() -> str | None:
    config = ROOT / "configs" / "lab.local.yaml"
    if not config.exists():
        return None
    return (yaml.safe_load(config.read_text()) or {}).get("robot", {}).get("follower_port")


def open_bus(port: str, calibration_path: Path):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    calibration = {
        name: MotorCalibration(**values)
        for name, values in json.loads(calibration_path.read_text()).items()
    }
    motors = {
        name: Motor(
            i,
            "sts3215",
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for i, name in enumerate(JOINT_NAMES, start=1)
    }
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    bus.connect()
    return bus, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=default_port(), help="Default: lab.local.yaml")
    parser.add_argument("--calibration", type=Path, default=default_calibration())
    parser.add_argument("--watch", type=float, default=0.0, help="Repeat for N seconds")
    args = parser.parse_args()
    if not args.port:
        raise SystemExit("No follower port; pass --port or set robot.follower_port")

    model = SO101Model()
    bus, calibration = open_bus(args.port, args.calibration)
    try:
        stored = bus.read_calibration()
        stale = [name for name in JOINT_NAMES if stored[name] != calibration[name]]
        if stale:
            print(f"WARNING motor registers differ from {args.calibration} for: {stale}")
        end = time.time() + args.watch
        while True:
            positions = bus.sync_read("Present_Position")
            xyz = model.fk_lerobot(positions)[:3, 3]
            joints = " ".join(f"{name}={positions[name]:7.1f}" for name in JOINT_NAMES)
            print(f"{joints} | tool xyz m {np.array2string(xyz, precision=3)}", flush=True)
            if time.time() >= end:
                break
            time.sleep(0.2)
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
