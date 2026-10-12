"""Printed calibration sheet: four tag36h11 AprilTags plus nine touch marks on landscape A4.

The arm touches the marks (``fk_fit``), which gives the joint offsets and the sheet's pose in the
robot base frame. The scene camera sees the tags, which gives the sheet's pose in the camera
frame. Together: ``T_base_camera = T_base_sheet @ inv(T_camera_sheet)``.

Sheet frame: origin at the page's lower-left corner, x along the long edge, y away from the robot
(the near row N is on the robot side), z up out of the paper; metres.
"""

import numpy as np
from numpy.typing import NDArray

PAGE_MM = (297.0, 210.0)  # A4 landscape
TAG_FAMILY = "tag36h11"
TAG_SIZE_MM = 45.0  # black square side
TAG_CELLS = 8  # 6x6 data bits + 1-cell black border
# Lower-left corner of each tag's black square on the page (mm). Ids 10-13 so they never clash
# with the gripper tags (0-4) from make_apriltags.py.
TAGS_MM = {10: (15.0, 15.0), 11: (237.0, 15.0), 12: (15.0, 150.0), 13: (237.0, 150.0)}
# Touch marks (mm): rows N/M/F away from the robot, columns L/C/R left to right as seen from
# behind the robot.
_XS = {"L": 80.0, "C": 148.5, "R": 217.0}
_YS = {"N": 35.0, "M": 105.0, "F": 175.0}
MARKS_MM = {row + col: (_XS[col], _YS[row]) for row in "NMF" for col in "LCR"}


def marks_m() -> dict[str, tuple[float, float]]:
    return {k: (x / 1000, y / 1000) for k, (x, y) in MARKS_MM.items()}


def tag_corners_m(tag_id: int) -> NDArray:
    """Tag corners in the sheet frame [4, 3], in OpenCV's detection order (as printed): top-left,
    top-right, bottom-right, bottom-left."""
    x, y = TAGS_MM[tag_id]
    s = TAG_SIZE_MM
    corners = [(x, y + s), (x + s, y + s), (x + s, y), (x, y)]
    return np.array([(cx / 1000, cy / 1000, 0.0) for cx, cy in corners])


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
        if int(i) in TAGS_MM
    }


def sheet_pose_in_camera(
    detections: dict[int, NDArray], k: NDArray, dist: NDArray | None = None
) -> tuple[NDArray, float]:
    """T_camera_sheet (4x4) from detected tag corners, and the RMS reprojection error (px)."""
    import cv2

    if len(detections) < 2:
        raise ValueError(f"Need at least 2 sheet tags, saw {sorted(detections)}")
    obj = np.vstack([tag_corners_m(i) for i in detections])
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
