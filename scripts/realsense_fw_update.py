"""Flash signed firmware to one RealSense D400 camera using pyrealsense2.

Default is a dry run: checks the image against the camera and saves a flash
backup (calibration + device info). Pass --flash to actually burn firmware.
"""

import argparse
import sys
import time
from pathlib import Path

import pyrealsense2 as rs


def find(serial: str):
    for dev in rs.context().query_devices():
        if (
            dev.supports(rs.camera_info.serial_number)
            and dev.get_info(rs.camera_info.serial_number) == serial
        ):
            return dev
    return None


def progress(label):
    last = [-1]

    def cb(p: float) -> None:
        pct = int(p * 100)
        if pct != last[0] and pct % 5 == 0:
            print(f"  {label} {pct}%", flush=True)
            last[0] = pct

    return cb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="Signed_Image_UVC_5_17_0_10.bin")
    ap.add_argument("--serial", default="819612070424")
    ap.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "calibration" / "local" / "realsense",
    )
    ap.add_argument("--flash", action="store_true", help="Actually burn the firmware")
    args = ap.parse_args()

    if not args.image.name.startswith("Signed_Image_UVC_"):
        sys.exit(f"Refusing {args.image.name}: expected a Signed_Image_UVC_*.bin file")
    image = list(args.image.read_bytes())

    dev = find(args.serial)
    if dev is None:
        sys.exit(f"No camera with serial {args.serial}")
    info = lambda k: dev.get_info(k) if dev.supports(k) else "?"  # noqa: E731
    firmware = info(rs.camera_info.firmware_version)
    print(f"{info(rs.camera_info.name)} {args.serial} firmware {firmware}")

    upd = dev.as_updatable()
    if not upd.check_firmware_compatibility(image):
        sys.exit("Camera reports this image is NOT compatible. Not flashing.")
    print("OK image is compatible with this camera")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = (
        args.backup_dir / f"d435_{args.serial}_flash_backup_{time.strftime('%Y%m%d-%H%M%S')}.bin"
    )
    backup.write_bytes(bytes(upd.create_flash_backup(progress("backup"))))
    print(f"OK flash backup saved to {backup}")

    if not args.flash:
        print("Dry run complete. Re-run with --flash to update.")
        return

    print("Entering update mode. Do NOT unplug the camera or let the laptop sleep.")
    upd.enter_update_state()
    deadline = time.time() + 30
    dfu = None
    while dfu is None and time.time() < deadline:
        time.sleep(1)
        dfu = next((d for d in rs.context().query_devices() if d.is_update_device()), None)
    if dfu is None:
        sys.exit("Camera did not reappear in update mode within 30 s. Replug and re-run.")
    dfu.as_update_device().update(image, progress("flash"))
    print("Flash written; waiting for the camera to reboot...")

    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(2)
        dev = find(args.serial)
        if dev is not None:
            print(f"OK camera back with firmware {dev.get_info(rs.camera_info.firmware_version)}")
            return
    sys.exit("Camera did not come back within 60 s; replug it and check rs.context().")


if __name__ == "__main__":
    main()
