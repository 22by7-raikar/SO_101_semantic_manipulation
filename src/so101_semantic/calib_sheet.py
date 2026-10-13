"""Printed calibration sheet: four tag36h11 AprilTags plus nine touch marks, landscape A4 or US
Letter (each printed at 100%; the layouts differ, so every step names the paper it used).

The arm touches the marks (``fk_fit``), which gives the joint offsets and the sheet's pose in the
robot base frame. The scene camera sees the tags, which gives the sheet's pose in the camera
frame. Together: ``T_base_camera = T_base_sheet @ inv(T_camera_sheet)``.

Sheet frame: origin at the page's lower-left corner, x along the long edge, y away from the robot
(the near row N is on the robot side), z up out of the paper; metres.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TAG_FAMILY = "tag36h11"
TAG_SIZE_MM = 45.0  # black square side
TAG_CELLS = 8  # 6x6 data bits + 1-cell black border


@dataclass(frozen=True)
class SheetLayout:
    """One printed layout, in mm on the page (origin at the lower-left corner, landscape)."""

    page_mm: tuple[float, float]
    tags_mm: dict[int, tuple[float, float]]  # lower-left corner of each tag's black square
    marks_mm: dict[str, tuple[float, float]]  # touch-mark centres


def _layout(page_mm, tag_x, xs, ys=(35.0, 105.0, 175.0), tag_y=(15.0, 150.0)) -> SheetLayout:
    """Tags ids 10-13 (never clashing with the gripper tags 0-4 from make_apriltags.py); marks in
    rows N/M/F away from the robot, columns L/C/R left to right seen from behind the robot."""
    left, right = tag_x
    near, far = tag_y
    tags = {10: (left, near), 11: (right, near), 12: (left, far), 13: (right, far)}
    marks = {
        r + c: (x, y)
        for r, y in zip("NMF", ys, strict=True)
        for c, x in zip("LCR", xs, strict=True)
    }
    return SheetLayout(page_mm, tags, marks)


LAYOUTS = {
    "a4": _layout((297.0, 210.0), tag_x=(15.0, 237.0), xs=(80.0, 148.5, 217.0)),
    # US Letter at 100%: same tags, rows and mark spacing, outer tags 12 mm from the edges.
    "letter": _layout((279.4, 215.9), tag_x=(12.0, 222.4), xs=(71.2, 139.7, 208.2)),
}
# The A4 layout under the original names.
PAGE_MM = LAYOUTS["a4"].page_mm
TAGS_MM = LAYOUTS["a4"].tags_mm
MARKS_MM = LAYOUTS["a4"].marks_mm


# A sheet printed with scaling (e.g. A4 shrunk to fit Letter) keeps the pattern: pass the measured
# per-axis scale (printed / designed), so every position becomes (sx * x, sy * y).
Scale = tuple[float, float]


def measured_scale(near_tags_mm: float, left_tags_mm: float, paper: str = "a4") -> Scale:
    """Scale from two ruler measurements on the print: left edge of tag 10 to left edge of tag 11
    (along the near row) and bottom edge of tag 10 to bottom edge of tag 12 (up the left side)."""
    tags = LAYOUTS[paper].tags_mm
    return (near_tags_mm / (tags[11][0] - tags[10][0]), left_tags_mm / (tags[12][1] - tags[10][1]))


def marks_m(paper: str = "a4", scale: Scale = (1.0, 1.0)) -> dict[str, tuple[float, float]]:
    sx, sy = scale
    return {k: (sx * x / 1000, sy * y / 1000) for k, (x, y) in LAYOUTS[paper].marks_mm.items()}


def tag_corners_m(tag_id: int, paper: str = "a4", scale: Scale = (1.0, 1.0)) -> NDArray:
    """Tag corners in the sheet frame [4, 3], in OpenCV's detection order (as printed): top-left,
    top-right, bottom-right, bottom-left."""
    x, y = LAYOUTS[paper].tags_mm[tag_id]
    s = TAG_SIZE_MM
    sx, sy = scale
    corners = [(x, y + s), (x + s, y + s), (x + s, y), (x, y)]
    return np.array([(sx * cx / 1000, sy * cy / 1000, 0.0) for cx, cy in corners])


def detector():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_sheet(image: NDArray) -> dict[int, NDArray]:
    """Detected sheet tags: id -> image corners [4, 2] (OpenCV order)."""
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector().detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(i): c.reshape(4, 2)
        for c, i in zip(corners, ids.ravel(), strict=True)
        if int(i) in TAGS_MM  # same ids in every layout
    }


def sheet_pose_in_camera(
    detections: dict[int, NDArray],
    k: NDArray,
    dist: NDArray | None = None,
    paper: str = "a4",
    scale: Scale = (1.0, 1.0),
) -> tuple[NDArray, float]:
    """T_camera_sheet (4x4) from detected tag corners, and the RMS reprojection error (px)."""
    import cv2

    if len(detections) < 2:
        raise ValueError(f"Need at least 2 sheet tags, saw {sorted(detections)}")
    obj = np.vstack([tag_corners_m(i, paper, scale) for i in detections])
    img = np.vstack([detections[i] for i in detections]).astype(np.float64)
    dist = np.zeros(5) if dist is None else dist
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    t[:3, 3] = tvec.ravel()
    return t, rms


def sheet_pose_in_base(x_m: float, y_m: float, yaw_deg: float, table_z_m: float) -> NDArray:
    """T_base_sheet (4x4) from the touch fit: the sheet lies flat on the table."""
    yaw = np.deg2rad(yaw_deg)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    t[:3, 3] = [x_m, y_m, table_z_m]
    return t


def base_to_camera(t_base_sheet: NDArray, t_camera_sheet: NDArray) -> NDArray:
    """T_base_camera (p_base = T @ p_camera)."""
    return t_base_sheet @ np.linalg.inv(t_camera_sheet)
