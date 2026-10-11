# Hardware bring-up: SO-101 leader/follower and RealSense D435

Run on the Ubuntu laptop with the `robot` profile installed ([environment setup](ENVIRONMENT.md)).
Interactive steps (calibration, teleoperation) need a local desktop terminal and one operator.
Keep measured values in `configs/lab.local.yaml` and calibration files in `calibration/local/`;
both are gitignored. Share calibration files with collaborators separately.

## 1. Connect and identify devices

- Each arm's controller board needs its own USB data cable. The follower also needs its motor
  supply (check it matches the STS3215 variant; the follower here reads ~12 V, the leader ~5 V).
- The D435 needs USB 3. Confirm with `lsusb -t` (5000M) and avoid USB 2 hubs.
- Arm boards appear as `1a86:55d3` and `/dev/serial/by-id/usb-1a86_USB_Single_Serial_<serial>-if00`.
  Use these by-id paths, not `/dev/ttyACM*`, in `lab.local.yaml`.
- If a device is missing, `journalctl -k --since -10min | grep -i usb` shows whether it ever
  enumerated. No entry usually means a charge-only cable or an unpowered board.

## 2. RealSense permissions, serial and firmware

Install the udev rules as in [ENVIRONMENT.md §4](ENVIRONMENT.md#4-realsense-and-so-101-host-permissions),
then replug the camera. The device node should become `crw-rw-rw- root plugdev`.

**Serial numbers:** the USB descriptor serial (shown by the kernel and `lsusb -v`) differs from
the camera serial that pyrealsense2 and LeRobot use. Read the latter with:

```bash
uv run --no-sync python -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.firmware_version)) for d in rs.context().query_devices()]"
```

**Firmware:** use the release matched to the SDK (`pyrealsense2==2.56.5` → D400 firmware
5.17.0.10, from the [D400 firmware releases](https://dev.realsenseai.com/docs/firmware-releases-d400)).
`scripts/realsense_fw_update.py` checks compatibility and saves a flash backup by default;
`--flash` burns the image. Connect the camera directly to the laptop and keep it plugged in
until the script reports the new version.

```bash
uv run --no-sync python scripts/realsense_fw_update.py <dir>/Signed_Image_UVC_5_17_0_10.bin --serial <camera serial>
uv run --no-sync python scripts/realsense_fw_update.py <dir>/Signed_Image_UVC_5_17_0_10.bin --serial <camera serial> --flash
```

Re-check `so101-doctor --camera`, depth scale and intrinsics afterwards.

## 3. Arm calibration

Motor IDs 1–6 must already be set (`lerobot-setup-motors` otherwise). Store calibration in the
repo's ignored directory:

```bash
export HF_LEROBOT_CALIBRATION=$PWD/calibration/local/lerobot
uv run --no-sync lerobot-calibrate --robot.type=so101_follower --robot.port=<follower by-id> --robot.id=follower
uv run --no-sync lerobot-calibrate --teleop.type=so101_leader --teleop.port=<leader by-id> --teleop.id=leader
```

Each run has two phases:

1. **Middle pose, then ENTER.** Every joint near the middle of its travel: pan forward, elbow
   about 90°, wrist in line with the forearm, wrist roll level, gripper half open. Hold the
   follower in place; with torque off it sags and the recorded centre is wrong. Pose both arms
   the same way, because teleop maps leader to follower through these centres.
2. **"Recording positions":** sweep each joint, including the gripper, end to end, then ENTER.

Before pressing ENTER in phase 2, check the live table. Arm joints typically span 2200–2700
counts and the gripper about 1200–1500. Abort (Ctrl-C) and redo if:

| Symptom | Cause |
| --- | --- |
| Gripper MIN ≈ MAX | Gripper not moved during phase 2 |
| A non-wrist-roll joint shows 0 / 4095 | Joint was far from centre in phase 1, so the encoder wrapped |
| `ValueError: Magnitude 2048 exceeds 2047` | A motor read exactly 4095 at ENTER; move it slightly and retry |

Calibration defines the joint values in datasets, policies, kinematics and `T_base_camera`.
Freeze it before recording data or calibrating the camera; recalibrating afterwards requires
redoing camera calibration and keeping datasets separate.

`uv run --no-sync so101-pose` reads the follower without writing to it and warns if the motor
registers differ from the calibration file. Do not run it while another process uses the port.

## 4. Teleoperation

Pose the arms alike before starting; the follower enables torque and moves to the leader.
Keep the follower's power within reach and support the arm before Ctrl-C (torque is released).

```bash
uv run --no-sync lerobot-teleoperate \
  --robot.type=so101_follower --robot.id=follower --robot.port=<follower by-id> \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader --teleop.id=leader --teleop.port=<leader by-id>
```

Add the cameras and Rerun display once the arms track (`wrist` and `scene`, as named in
`lab.local.yaml`; both at the 640×480, 30 fps the policy uses):

```bash
  --robot.cameras="{ wrist: {type: opencv, index_or_path: '<wrist /dev/v4l/by-id path>', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, scene: {type: intelrealsense, serial_number_or_name: '<camera serial>', width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

Only one process can open each camera: stop any other viewer or stream first.

To look at the cameras without teleoperating (e.g. when mounting or placing them):

```bash
uv run --no-sync python scripts/stream_cameras.py                   # scene + wrist
uv run --no-sync python scripts/stream_cameras.py --cameras wrist --guides
```

## 5. Kinematic model check

`third_party/so101/so101_new_calib.urdf` is pinned upstream (see its README). LeRobot's degree
readings use the same zero (middle of the calibrated range), but each joint's direction and the
provisional gripper mapping must be confirmed on the real arm before the model is used for
camera calibration or planning:

1. Run `uv run --no-sync so101-pose --watch 60` with the follower torque off.
2. Move one joint at a time and check the reported tool position moves the same way as the
   real gripper (for example, lifting the arm increases z; +x points forward from the base).
3. In two or three poses, measure the gripper tip from the base with a ruler and compare.
4. Record any reversed joint in `JOINT_SIGNS` in `src/so101_semantic/kinematics.py`, with the
   evidence, and only then set `robot.urdf_path` as verified in `lab.local.yaml`.
