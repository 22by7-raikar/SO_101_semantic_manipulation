"""Write a printable PDF of tag36h11 AprilTags at exact physical sizes, with a scale bar.

  uv run --no-sync python scripts/make_apriltags.py            # -> outputs/apriltags_tag36h11.pdf

Print at 100% scale ("actual size", never "fit to page") on matte paper, then measure the black
square: the calibration needs its true side length. Sizes are the black square's side (the
marker length OpenCV expects); each tag keeps a one-cell white border and a light cut line.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MM_PER_INCH = 25.4
PAGE_MM = (215.9, 279.4)  # US Letter; content stays inside A4 too (210 x 297 mm)
CELLS = 8  # tag36h11: 6x6 data bits + 1-cell black border


def draw_tag(ax, tag_id: int, side_mm: float, x_mm: float, y_mm: float) -> None:
    """Place one tag with its black square's lower-left corner at (x_mm, y_mm)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = cv2.aruco.generateImageMarker(dictionary, tag_id, CELLS * 50, borderBits=1)
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        extent=(x_mm, x_mm + side_mm, y_mm, y_mm + side_mm),
    )
    cell = side_mm / CELLS
    ax.add_patch(
        Rectangle(
            (x_mm - cell, y_mm - cell),
            side_mm + 2 * cell,
            side_mm + 2 * cell,
            fill=False,
            edgecolor="0.75",
            linestyle="--",
            linewidth=0.4,
        )
    )
    ax.text(
        x_mm + side_mm / 2,
        y_mm - cell - 3,
        f"tag36h11 id {tag_id} · black square {side_mm:.1f} mm",
        ha="center",
        va="top",
        fontsize=6.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "apriltags_tag36h11.pdf")
    parser.add_argument("--png-dpi", type=int, help="Also write a PNG at this DPI (for checking)")
    args = parser.parse_args()

    w, h = PAGE_MM
    fig = plt.figure(figsize=(w / MM_PER_INCH, h / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        w / 2, h - 14, "AprilTags tag36h11 for camera-to-base calibration", ha="center", fontsize=11
    )
    ax.text(
        w / 2,
        h - 21,
        "Print at 100% / actual size (not fit to page). Measure the black square; "
        "glue flat on rigid backing. Use one tag on the fixed gripper body.",
        ha="center",
        fontsize=7,
    )

    # Row 1: three 50 mm tags; row 2: a 50 mm and a 60 mm tag. Cut lines never overlap.
    for i, x in enumerate((17.0, 82.0, 147.0)):
        draw_tag(ax, i, 50.0, x, 185.0)
    draw_tag(ax, 3, 50.0, 30.0, 95.0)
    draw_tag(ax, 4, 60.0, 115.0, 95.0)

    # 100 mm scale bar with 10 mm ticks to check the print scale.
    x0, y0 = 20.0, 45.0
    ax.plot([x0, x0 + 100], [y0, y0], color="k", linewidth=1.2)
    for t in range(11):
        tick = 4 if t % 5 == 0 else 2
        ax.plot([x0 + 10 * t, x0 + 10 * t], [y0, y0 + tick], color="k", linewidth=0.8)
    ax.text(
        x0 + 50, y0 - 4, "this bar must measure exactly 100 mm", ha="center", va="top", fontsize=7
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Wrote {args.out}")
    if args.png_dpi:
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=args.png_dpi)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
