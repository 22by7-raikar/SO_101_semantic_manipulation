"""Stream the lab cameras into a Rerun viewer, using the settings in configs/lab.local.yaml.

  uv run --no-sync python scripts/stream_cameras.py                  # all configured cameras
  uv run --no-sync python scripts/stream_cameras.py --cameras wrist --guides

scene (RealSense): policy colour, filtered depth aligned to colour (shown as a point cloud in the
3D view) and raw depth for comparison. wrist (USB): the policy image, with optional centre and
thirds guides for mounting. Only one process can open a camera, so stop teleop or recording
first. Ctrl-C (or SIGTERM) releases the cameras. Read-only: never touches the motors.
"""

import argparse
import signal
import threading
import time
from pathlib import Path

import numpy as np
import rerun as rr
import yaml

ROOT = Path(__file__).resolve().parents[1]


def stream_scene(cfg: dict, stop: threading.Event, temporal: bool) -> None:
    import pyrealsense2 as rs

    colour, depth = cfg["color"], cfg["depth"]
    rs_cfg = rs.config()
    rs_cfg.enable_device(str(cfg["serial"]))
    rs_cfg.enable_stream(
        rs.stream.color, colour["width"], colour["height"], rs.format.rgb8, colour["fps"]
    )
    rs_cfg.enable_stream(
        rs.stream.depth, depth["width"], depth["height"], rs.format.z16, depth["fps"]
    )
    # A fresh start occasionally delivers no frames (seen right after another process released
    # the camera); restarting the pipeline clears it.
    for attempt in range(3):
        pipe = rs.pipeline()
        profile = pipe.start(rs_cfg)
        if pipe.try_wait_for_frames(3000)[0]:
            break
        pipe.stop()
        print(f"scene: no frames on start, retrying ({attempt + 1}/3)")
    else:
        raise RuntimeError("scene camera started but delivered no frames")
    try:
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if depth.get("emitter", True) else 0)
        scale = sensor.get_depth_scale()
        # RealSense's recommended order: smooth in disparity space, back to depth, then align.
        filters = [rs.disparity_transform(True)]
        if depth.get("spatial_filter", True):
            filters.append(rs.spatial_filter())
        if temporal:
            filters.append(rs.temporal_filter())
        filters.append(rs.disparity_transform(False))
        align = rs.align(rs.stream.color)
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        rr.log(
            "scene",
            rr.Pinhole(
                focal_length=[k.fx, k.fy],
                principal_point=[k.ppx, k.ppy],
                width=k.width,
                height=k.height,
            ),
            static=True,
        )
        print(f"scene: colour {k.width}x{k.height}, depth {depth['width']}x{depth['height']}")
        while not stop.is_set():
            frames = pipe.wait_for_frames()
            raw = np.asanyarray(frames.get_depth_frame().get_data())
            frameset = frames
            for f in filters:
                frameset = f.process(frameset).as_frameset()
            frameset = align.process(frameset)
            rr.set_time("time", timestamp=time.time())
            rr.log(
                "scene/color",
                rr.Image(np.asanyarray(frameset.get_color_frame().get_data())).compress(
                    jpeg_quality=90
                ),
            )
            filtered = np.asanyarray(frameset.get_depth_frame().get_data())
            rr.log("scene/depth", rr.DepthImage(filtered, meter=1 / scale))
            rr.log("scene_raw_depth", rr.DepthImage(raw, meter=1 / scale))
    finally:
        pipe.stop()
        print("scene: released")


def stream_wrist(cfg: dict, stop: threading.Event, guides: bool) -> None:
    import cv2

    cap = cv2.VideoCapture(str(cfg["path"]), cv2.CAP_V4L2)
    try:
        if cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open wrist camera {cfg['path']}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if guides:
            centre = [[[w / 2, 0], [w / 2, h]], [[0, h / 2], [w, h / 2]]]
            thirds = [[[x, 0], [x, h]] for x in (w / 3, 2 * w / 3)]
            thirds += [[[0, y], [w, y]] for y in (h / 3, 2 * h / 3)]
            rr.log("wrist/guide_centre", rr.LineStrips2D(centre, colors=[255, 60, 60]), static=True)
            rr.log(
                "wrist/guide_thirds",
                rr.LineStrips2D(thirds, colors=[255, 255, 255, 90]),
                static=True,
            )
        print(f"wrist: {w}x{h}")
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            rr.set_time("time", timestamp=time.time())
            rr.log(
                "wrist/image",
                rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=90),
            )
    finally:
        cap.release()
        print("wrist: released")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lab.local.yaml")
    parser.add_argument("--cameras", nargs="+", choices=["scene", "wrist"], help="Default: all")
    parser.add_argument("--guides", action="store_true", help="Mounting guides on the wrist image")
    parser.add_argument(
        "--temporal", action="store_true", help="Temporal depth filter (smears moving objects)"
    )
    parser.add_argument("--save", type=Path, help="Write a .rrd file instead of opening a viewer")
    parser.add_argument("--seconds", type=float, help="Stop after this long")
    args = parser.parse_args()

    cameras = (yaml.safe_load(args.config.read_text()) or {}).get("cameras") or {}
    names = args.cameras or [n for n in ("scene", "wrist") if n in cameras]
    missing = [n for n in names if n not in cameras]
    if missing or not names:
        raise SystemExit(f"No config for cameras {missing or names} in {args.config}")

    rr.init("so101_cameras", spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    errors = []

    def run(target, *a):
        try:
            target(*a)
        except Exception as e:  # report and stop the other camera too
            errors.append(e)
            stop.set()

    workers = {
        "scene": lambda: run(stream_scene, cameras["scene"], stop, args.temporal),
        "wrist": lambda: run(stream_wrist, cameras["wrist"], stop, args.guides),
    }
    threads = [threading.Thread(target=workers[n], daemon=True) for n in names]
    for t in threads:
        t.start()
    end = time.time() + args.seconds if args.seconds else None
    try:
        while not stop.is_set() and (end is None or time.time() < end):
            stop.wait(0.2)
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=5)
    if errors:
        raise SystemExit(f"Stopped: {errors[0]}")


if __name__ == "__main__":
    main()
