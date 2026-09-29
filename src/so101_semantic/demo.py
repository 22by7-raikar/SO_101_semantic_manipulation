"""Synthetic integration artifact; no camera, model download, or robot motion."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from so101_semantic.geometry import backproject, transform_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Synthetic intrinsics and transform, never usable as lab calibration.
    depth = np.full((32, 32), 0.4)
    k = np.array([[100, 0, 15.5], [0, 100, 15.5], [0, 0, 1]])
    points = transform_points(backproject(depth, k), np.eye(4))
    origin = np.array([-0.1, -0.1, 0.0], dtype=np.float32)
    voxel_size = 0.01
    shape = (20, 20, 60)
    costs = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    indices = np.floor((points - origin) / voxel_size).astype(int)
    inside = ((indices >= 0) & (indices < np.array(shape))).all(axis=1)
    indices = indices[inside]
    # Synthetic semantic penalty; unobserved zero cells are NOT proven free space.
    penalty = np.exp(-np.sum(points[inside, :2] ** 2, axis=1) / 0.002)
    np.maximum.at(costs, tuple(indices.T), penalty.astype(np.float32))
    observed[tuple(indices.T)] = True
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "cost_field.npz",
        costs=costs,
        observed=observed,
        origin_m=origin,
        voxel_size_m=np.float32(voxel_size),
        frame="synthetic_base",
        axis_order="xyz",
        schema_version=np.int64(1),
    )
    fig, ax = plt.subplots()
    ax.imshow(costs.max(axis=2).T, origin="lower", extent=(-0.1, 0.1, -0.1, 0.1))
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Synthetic cost field (max over z)")
    fig.savefig(args.output / "cost_field.png", dpi=150)
    plt.close(fig)
    print(f"Wrote synthetic field and preview to {args.output}")


if __name__ == "__main__":
    main()
