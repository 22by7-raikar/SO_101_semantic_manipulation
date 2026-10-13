import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from so101_semantic.calib_sheet import (
    LAYOUTS,
    TAG_CELLS,
    TAG_SIZE_MM,
    TAGS_MM,
    base_to_camera,
    detect_sheet,
    marks_m,
    sheet_pose_in_base,
    sheet_pose_in_camera,
    tag_corners_m,
)

K = np.array([[920.0, 0, 640.0], [0, 920.0, 360.0], [0, 0, 1.0]])


@pytest.mark.parametrize("paper", sorted(LAYOUTS))
def test_layout_fits_the_page_without_overlaps(paper):
    layout = LAYOUTS[paper]
    w, h = layout.page_mm
    border = TAG_SIZE_MM / TAG_CELLS  # white quiet zone around each tag
    boxes = [
        (x - border, y - border, x + TAG_SIZE_MM + border, y + TAG_SIZE_MM + border)
        for x, y in layout.tags_mm.values()
    ]
    for x0, y0, x1, y1 in boxes:
        assert x0 >= 6.35 and y0 >= 6.35 and x1 <= w - 6.35 and y1 <= h - 6.35  # 1/4 in margins
    for x, y in layout.marks_mm.values():
        assert all(not (x0 - 8 < x < x1 + 8 and y0 - 8 < y < y1 + 8) for x0, y0, x1, y1 in boxes)
    assert marks_m(paper)["MC"] == pytest.approx((w / 2000, 0.105))
    assert sorted(layout.tags_mm) == sorted(TAGS_MM)


def test_letter_keeps_the_mark_pattern():
    a4, letter = (np.array(list(LAYOUTS[p].marks_mm.values())) for p in ("a4", "letter"))
    np.testing.assert_allclose(letter - letter.mean(0), a4 - a4.mean(0), atol=1e-9)


def camera_looking_at_sheet():
    """T_camera_sheet for a camera ~0.7 m away, looking down at the sheet from the far side."""
    rvec = np.array([2.3, 0.1, -0.15])
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    t[:3, 3] = [-0.15, 0.05, 0.7]
    return t


def project(t_camera_sheet, tag_id):
    rvec = cv2.Rodrigues(t_camera_sheet[:3, :3])[0]
    pts, _ = cv2.projectPoints(tag_corners_m(tag_id), rvec, t_camera_sheet[:3, 3], K, np.zeros(5))
    return pts.reshape(4, 2)


def test_sheet_pose_round_trip():
    truth = camera_looking_at_sheet()
    detections = {i: project(truth, i) for i in TAGS_MM}
    estimate, rms = sheet_pose_in_camera(detections, K)
    np.testing.assert_allclose(estimate, truth, atol=1e-6)
    assert rms < 1e-3


def test_sheet_pose_with_corner_noise_and_a_missing_tag():
    truth = camera_looking_at_sheet()
    rng = np.random.default_rng(0)
    detections = {i: project(truth, i) + rng.normal(scale=0.3, size=(4, 2)) for i in (10, 11, 13)}
    estimate, _ = sheet_pose_in_camera(detections, K)
    assert np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) < 0.003  # metres


def test_base_to_camera_composition():
    t_base_sheet = sheet_pose_in_base(0.12, -0.15, 87.0, -0.0024)
    t_camera_sheet = camera_looking_at_sheet()
    t_base_camera = base_to_camera(t_base_sheet, t_camera_sheet)
    p_sheet = np.array([0.1485, 0.105, 0.0, 1.0])  # mark MC
    p_camera = t_camera_sheet @ p_sheet
    np.testing.assert_allclose(t_base_camera @ p_camera, t_base_sheet @ p_sheet, atol=1e-12)


@pytest.mark.parametrize("paper", sorted(LAYOUTS))
def test_rendered_sheet_detects_with_matching_corners(tmp_path, paper):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_calib_sheet import draw

    png = tmp_path / "sheet.png"
    dpi = 150
    draw(png, paper, dpi=dpi)
    detections = detect_sheet(cv2.imread(str(png)))
    assert sorted(detections) == sorted(TAGS_MM)
    px_per_m = dpi / 0.0254
    for tag_id, corners in detections.items():
        expected = tag_corners_m(tag_id, paper)[:, :2] * px_per_m
        expected[:, 1] = LAYOUTS[paper].page_mm[1] / 1000 * px_per_m - expected[:, 1]  # y down
        assert np.abs(corners - expected).max() < 2.0, tag_id  # pixels (~0.3 mm)
