"""Write the printable calibration sheet (A4 landscape): four AprilTags and nine touch marks.

  uv run --no-sync python scripts/make_calib_sheet.py        # -> outputs/calib_sheet_a4.pdf

Print at 100% / actual size on matte paper; check the 100 mm scale bar and a tag's 45 mm black
square. Tape it flat in front of the arm with the N (near) row toward the robot base.
Geometry lives in so101_semantic.calib_sheet, shared with the fit and the camera step.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from so101_semantic.calib_sheet import (  # noqa: E402
    MARKS_MM,
    PAGE_MM,
    TAG_CELLS,
    TAG_SIZE_MM,
    TAGS_MM,
)

ROOT = Path(__file__).resolve().parents[1]
MM_PER_INCH = 25.4


def draw(path: Path, dpi: int | None = None) -> None:
    w, h = PAGE_MM
    fig = plt.figure(figsize=(w / MM_PER_INCH, h / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id, (x, y) in TAGS_MM.items():
        image = cv2.aruco.generateImageMarker(dictionary, tag_id, TAG_CELLS * 50, borderBits=1)
        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            extent=(x, x + TAG_SIZE_MM, y, y + TAG_SIZE_MM),
        )
        label_y = y - 4 if y < h / 2 else y + TAG_SIZE_MM + 2
        ax.text(x + TAG_SIZE_MM / 2, label_y, f"id {tag_id}", ha="center", fontsize=5, color="0.5")

    for label, (x, y) in MARKS_MM.items():
        ax.plot([x - 6, x + 6], [y, y], color="k", linewidth=0.5)
        ax.plot([x, x], [y - 6, y + 6], color="k", linewidth=0.5)
        ax.add_patch(Circle((x, y), 2.0, fill=False, edgecolor="k", linewidth=0.5))
        ax.text(x + 4, y + 3, label, fontsize=9, fontweight="bold")

    ax.text(w / 2, h - 8, "SO-101 calibration sheet (A4, print at 100%)", ha="center", fontsize=8)
    ax.text(
        w / 2,
        h - 14,
        "Tape flat; N row toward the robot base. Touch crosshair centres with the fixed-jaw point.",
        ha="center",
        fontsize=6,
    )
    x0, y0 = w / 2 - 50, 14.0
    ax.plot([x0, x0 + 100], [y0, y0], color="k", linewidth=1.0)
    for t in range(11):
        ax.plot(
            [x0 + 10 * t, x0 + 10 * t], [y0, y0 + (3 if t % 5 == 0 else 1.5)], color="k", lw=0.6
        )
    ax.text(w / 2, y0 - 2.5, "100 mm  |  robot base this side", ha="center", va="top", fontsize=6)
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor="0.85", linewidth=0.3))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "calib_sheet_a4.pdf")
    args = parser.parse_args()
    draw(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
