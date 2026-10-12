# Calibration steps: joint offsets, FK check and camera-to-base

One printed sheet does all three. Full background: [HARDWARE_SETUP.md §5](HARDWARE_SETUP.md#5-kinematic-model-check).
Run commands from the repo root. New terminals already set `HF_LEROBOT_CALIBRATION` (`~/.bashrc`).

## 1. Print and place the sheet

1. Print `outputs/calib_sheet_a4.pdf` at **100% / Actual size** (never "Fit to page"), A4, matte.
   Regenerate it with `uv run --no-sync python scripts/make_calib_sheet.py`.
2. Measure: the scale bar must be **100 mm**, a tag's black square **45 mm**.
3. Tape it **flat** in front of the follower: **N row toward the base**, near edge **3–5 cm** from
   the base, roughly centred on the direction the arm points when straight.
4. Tape all 4 corners and **mark the corners on the table**, so the sheet can go back exactly.
5. Route the wrist camera cable around the sheet, with slack for the far corners.

Marks, as seen from behind the robot facing the way it faces:

```
far row:    FL  FC  FR
middle row: ML  MC  MR
near row:   NL  NC  NR      <- robot base
```

## 2. Touch session (~15 min): joint offsets + FK check

1. Stop teleop (torque must be off). Close the gripper; hold or tape it closed.
2. Start the recorder:
   ```bash
   uv run --no-sync python scripts/record_touch_points.py
   ```
3. For each touch: support the arm, rest the **point of the fixed jaw** (the jaw that does not
   move when the gripper opens) lightly on a crosshair centre, type its name (e.g. `MC`), ENTER.
4. Touches (~16): all 9 marks once, then MC with the wrist rolled ~45° each way, NC and FC with the
   elbow bent more, ML / MR / NR with the wrist rolled. Skip marks the arm cannot reach.
   `u` = undo last, `q` = finish. The file is saved after every touch.
5. Fit and save:
   ```bash
   uv run --no-sync python -m so101_semantic.fk_fit calibration/local/fk_touch/<file>.json --save
   ```
   Check: residual RMS **< 10 mm**; note the offsets. Applying them in `kinematics.py` is a
   separate, reviewed step.

**Leave the sheet taped where it is** until step 3 is done.

## 3. Camera-to-base calibration (after the camera stand arrives)

1. Clamp the RealSense in its **final** position (arm, all block and bowl spots in view); mark it.
2. Make sure at least **3 of the sheet's 4 tags** are visible (check with
   `uv run --no-sync python scripts/stream_cameras.py --cameras scene`, then close it).
3. Run:
   ```bash
   uv run --no-sync python scripts/calibrate_scene_camera.py calibration/local/fk_touch/<file>.fit.json --write
   ```
   Check: reprojection RMS **< 1 px**, jitter **< 2 mm**. This writes `calibration.T_base_camera`
   to `configs/lab.local.yaml`.
4. The sheet can now come off the table (keep the corner marks).

## Redo when

| Change | Redo |
| --- | --- |
| Scene camera moved or bumped | Sheet back on its marks → step 3 only |
| Arm recalibrated (`lerobot-calibrate`) | Steps 2 and 3 |
| Sheet moved before step 3 | Step 2 again (new touches), then step 3 |

## If something goes wrong

- **LeRobot asks about a calibration mismatch:** press ENTER (use the files), never `c`.
- **Recorder says torque is on:** teleop is still running; stop it.
- **Recorder says motor calibration differs:** the arms were recalibrated elsewhere; ask before going on.
- **Camera busy:** close any stream, viewer or teleop using it.
