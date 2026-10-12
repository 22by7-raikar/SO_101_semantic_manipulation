"""Camera-to-base calibration from the printed sheet.

T_base_camera = T_base_sheet @ inv(T_camera_sheet).

T_base_sheet comes from the touch fit (fk_fit --save); T_camera_sheet from the sheet's four tags
seen by the scene RealSense. The sheet must not have moved since the touches. The camera frame
is the colour sensor's optical frame (x right, y down, z forward), the frame the policy and depth
alignment use; it is the same at every colour resolution, so frames are captured at 1280x720 for
detection accuracy.

  uv run --no-sync python scripts/calibrate_scene_camera.py \
      calibration/local/fk_touch/<file>.fit.json
  ... --write    # also store it in configs/lab.local.yaml (calibration.T_base_camera)

Read-only on the camera; never touches the motors.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from so101_semantic.calib_sheet import (
    base_to_camera,
    detect_sheet,
    sheet_pose_in_base,
    sheet_pose_in_camera,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lab.local.yaml"


def capture(serial: str, n_frames: int) -> tuple[list[np.ndarray], np.ndarray]:
    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    for _ in range(3):  # a fresh start occasionally delivers no frames
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        if pipe.try_wait_for_frames(3000)[0]:
            break
        pipe.stop()
    else:
        raise RuntimeError("scene camera delivered no frames")
    try:
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        kmat = np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1.0]])
        for _ in range(30):  # let auto-exposure settle
            pipe.wait_for_frames()
        images = [
            np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
            for _ in range(n_frames)
        ]
    finally:
        pipe.stop()
    return images, kmat


def write_to_config(t: np.ndarray, note: str) -> None:
    lines = CONFIG.read_text().splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip().startswith("T_base_camera:"))
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
    block = [f"{indent}T_base_camera:  # p_base = T_base_camera @ p_camera (scene colour); {note}"]
    block += [f"{indent}  - [{', '.join(f'{v:.6f}' for v in row)}]" for row in t]
    end = idx + 1
    while end < len(lines) and lines[end].startswith(indent + "  - "):
        end += 1  # replace an earlier matrix
    CONFIG.write_text("\n".join(lines[:idx] + block + lines[end:]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fit_file", type=Path, help="<touch file>.fit.json from fk_fit --save")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--write", action="store_true", help="Store in configs/lab.local.yaml")
    args = parser.parse_args()

    fit = json.loads(args.fit_file.read_text())
    if fit.get("paper") != "calib":
        raise SystemExit("The fit must come from touches on the printed calibration sheet")
    pose = fit["sheet_pose"]
    t_base_sheet = sheet_pose_in_base(pose["x_m"], pose["y_m"], pose["yaw_deg"], fit["table_z_m"])

    serial = str(yaml.safe_load(CONFIG.read_text())["cameras"]["scene"]["serial"])
    images, kmat = capture(serial, args.frames)
    poses, errors, seen = [], [], set()
    for image in images:
        detections = detect_sheet(image)
        if len(detections) >= 3:
            t, rms = sheet_pose_in_camera(detections, kmat)
            poses.append(t)
            errors.append(rms)
            seen |= set(detections)
    if not poses:
        raise SystemExit(
            "Sheet tags not found (need 3 of 4 in view); check the camera sees the sheet"
        )
    translations = np.array([p[:3, 3] for p in poses])
    t_camera_sheet = poses[
        int(np.argmin(np.linalg.norm(translations - np.median(translations, 0), axis=1)))
    ]
    t_base_camera = base_to_camera(t_base_sheet, t_camera_sheet)

    jitter_mm = np.ptp(translations, axis=0).max() * 1000
    print(
        f"tags seen {sorted(seen)} in {len(poses)}/{len(images)} frames; "
        f"reprojection RMS {np.median(errors):.2f} px; frame-to-frame jitter {jitter_mm:.1f} mm"
    )
    print(
        f"touch fit RMS {fit['rms_m'] * 1000:.1f} mm (sheet pose in base from {fit['touch_file']})"
    )
    print(f"camera position in base frame (m): {np.round(t_base_camera[:3, 3], 4).tolist()}")
    print("T_base_camera:\n" + np.array2string(t_base_camera, precision=4, suppress_small=True))

    out = args.fit_file.with_name(args.fit_file.name.replace(".fit.json", ".T_base_camera.json"))
    out.write_text(
        json.dumps(
            {
                "T_base_camera": t_base_camera.tolist(),
                "T_camera_sheet": t_camera_sheet.tolist(),
                "T_base_sheet": t_base_sheet.tolist(),
                "fit_file": str(args.fit_file),
                "reprojection_rms_px": float(np.median(errors)),
                "jitter_mm": float(jitter_mm),
                "time": time.time(),
            },
            indent=1,
        )
    )
    print(f"Saved {out}")
    if args.write:
        write_to_config(t_base_camera, f"from {out.name}")
        print(f"Updated {CONFIG}")


if __name__ == "__main__":
    main()
