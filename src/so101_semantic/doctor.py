"""Read-only environment diagnostics; never open a motor port or move a robot."""

import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", action="store_true", help="Check vision library imports")
    parser.add_argument("--gpu", action="store_true", help="Require working CUDA allocation")
    parser.add_argument("--hardware", action="store_true", help="Check hardware library imports")
    parser.add_argument("--camera", action="store_true", help="Enumerate RealSense devices")
    args = parser.parse_args()
    failures = []
    print(f"Python {platform.python_version()} | {platform.platform()} | {sys.executable}")
    if sys.version_info[:2] != (3, 11):
        failures.append("Use the repository's Python 3.11 environment")
    modules = ["numpy", "scipy", "cv2", "mujoco", "yourdfpy"]
    if args.vision or args.gpu:
        modules += ["torch", "torchvision", "transformers"]
    if args.hardware or args.camera:
        modules += ["pyrealsense2", "serial", "lerobot", "av", "torchcodec"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            print(f"OK import {name} {getattr(module, '__version__', '')}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if args.gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable; inspect nvidia-smi and the installed torch build"
                )
            x = torch.ones((32, 32), device="cuda")
            _ = (x @ x).sum().item()
            print(f"OK CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
            subprocess.run(["nvidia-smi"], check=True, timeout=15)
        except Exception as exc:
            failures.append(f"GPU: {exc}")
    if args.hardware:
        for package in ("lerobot", "feetech-servo-sdk", "pyrealsense2"):
            try:
                print(f"OK package {package} {importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                failures.append(f"Missing {package}")
    if args.camera:
        try:
            import pyrealsense2 as rs

            devices = rs.context().query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense detected; check USB 3 cable and udev permissions")
            print(f"OK RealSense: {len(devices)} device(s)")
        except Exception as exc:
            failures.append(f"Camera: {exc}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print("Requested checks passed. Calibration, collision checking and motor safety are untested.")


if __name__ == "__main__":
    main()
