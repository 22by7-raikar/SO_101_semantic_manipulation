"""Signed distance field from an obstacle point cloud, and arm clearance against it.

Input (the CV interface): a masked obstacle cloud in the robot base frame, N x 3 metres, with a
timestamp. The arm and the grasp target must already be removed from it (the gripper touches the
target, and its own points would read as collisions); whether the table is in it is the caller's
choice, but grasps bring the jaws to the table, so leave it out and bound the table separately.

The field covers an axis-aligned workspace box in voxels (1 cm by default). Cell ``[i, j, k]``
centre = ``origin_m + voxel_size_m * ([i, j, k] + 0.5)``, as in the cost-field NPZ contract.
Distances come from ``scipy.ndimage.distance_transform_edt`` between voxel centres, shifted by
half a voxel so the zero level sits on the occupied cells' faces: positive in free space, negative
inside occupied cells. A point occupies its whole voxel, so the field reads 0 to 1 voxel closer
than the true surface (conservative) along an axis, a little more along diagonals (the EDT runs
between voxel centres). A cloud of surfaces only (depth) leaves object interiors free: inside
values are negative only for filled occupancy (``fill_down`` or ``from_occupancy``).

Queries interpolate trilinearly between voxel centres (the same in numpy and torch, so values and
gradients agree). Outside the box a query clamps to the nearest boundary value, and the gradient
component along a clamped axis is zero: the box edge is unobserved, so the execution loop's
workspace box must keep the arm inside it.

Unknown space (occluded, out of view) is treated as free here; the camera sees obstacle tops, so
``fill_down`` marks every voxel below an occupied one as occupied, the conservative choice for
objects standing on the table.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from so101_semantic.kinematics import JOINT_NAMES

# base_link spheres extend below the table; use them for masking only (CLAUDE.md).
ENV_EXCLUDED_LINKS = ("base_link",)


@dataclass(frozen=True)
class DistanceField:
    distance: NDArray  # float32 [nx, ny, nz], metres, signed (> 0 free)
    origin_m: NDArray  # lower corner of the box, base frame
    voxel_size_m: float
    occupied: NDArray  # bool [nx, ny, nz]
    frame: str = "robot_base"
    stamp: float | None = None  # capture time of the source cloud, seconds

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.distance.shape

    @classmethod
    def from_points(
        cls,
        points: NDArray,
        box_min_m: NDArray,
        box_max_m: NDArray,
        voxel_size_m: float = 0.01,
        *,
        fill_down: bool = False,
        min_points: int = 1,
        frame: str = "robot_base",
        stamp: float | None = None,
    ) -> "DistanceField":
        """Voxelise an N x 3 obstacle cloud (base frame, metres) in the box; points outside it are
        dropped. Voxels with fewer than ``min_points`` points stay free (depth speckle)."""
        origin, shape = box_grid(box_min_m, box_max_m, voxel_size_m)
        occupied = occupancy(points, origin, shape, voxel_size_m, min_points)
        if fill_down:
            # Everything at or below the highest occupied voxel of each column.
            occupied = np.flip(np.logical_or.accumulate(np.flip(occupied, 2), axis=2), 2)
        return cls.from_occupancy(occupied, origin, voxel_size_m, frame=frame, stamp=stamp)

    @classmethod
    def from_occupancy(
        cls,
        occupied: NDArray,
        origin_m: NDArray,
        voxel_size_m: float,
        *,
        frame: str = "robot_base",
        stamp: float | None = None,
    ) -> "DistanceField":
        occupied = np.asarray(occupied, dtype=bool)
        if occupied.ndim != 3 or min(occupied.shape) < 2:
            raise ValueError(f"Need a 3-D grid at least 2 cells per axis, got {occupied.shape}")
        v = float(voxel_size_m)
        if not occupied.any():
            # No obstacles: every query is the box diagonal away (finite, flat).
            distance = np.full(occupied.shape, v * np.linalg.norm(occupied.shape), np.float32)
        else:
            outside = distance_transform_edt(~occupied, sampling=v) - 0.5 * v
            if occupied.all():
                inside = np.full(occupied.shape, -0.5 * v)
            else:
                inside = -(distance_transform_edt(occupied, sampling=v) - 0.5 * v)
            distance = np.where(occupied, inside, outside).astype(np.float32)
        return cls(distance, np.asarray(origin_m, float), v, occupied, frame, stamp)

    def _index(self, points: NDArray) -> NDArray:
        """Continuous voxel-centre index of each point [n, 3]."""
        return (np.asarray(points, float) - self.origin_m) / self.voxel_size_m - 0.5

    def query(self, points: NDArray) -> tuple[NDArray, NDArray]:
        """Distance [n] (metres) and its gradient [n, 3] at base-frame points [n, 3]."""
        points = np.atleast_2d(points)
        value, grad = _trilinear(self.distance, self._index(points))
        return value, grad / self.voxel_size_m

    def torch(self, device=None, dtype=None) -> "TorchDistanceField":
        return TorchDistanceField(self, device=device, dtype=dtype)


class TorchDistanceField:
    """Differentiable query (``grid_sample``, trilinear, border clamp); gradients via autograd."""

    def __init__(self, field: DistanceField, device=None, dtype=None) -> None:
        import torch

        dtype = dtype or torch.float32
        # grid_sample reads [N, C, D, H, W] with grid (x->W, y->H, z->D); store [nx, ny, nz]
        # as D, H, W and pass coordinates as (k, j, i).
        self.grid = torch.as_tensor(field.distance, dtype=dtype, device=device)[None, None]
        self.origin = torch.as_tensor(field.origin_m, dtype=dtype, device=device)
        self.voxel = field.voxel_size_m
        self.size = torch.as_tensor(field.shape, dtype=dtype, device=device)

    def __call__(self, points):
        """Distance at base-frame points [..., 3] (metres), same leading shape."""
        import torch.nn.functional as F

        lead = points.shape[:-1]
        index = (points.reshape(-1, 3).to(self.grid.dtype) - self.origin) / self.voxel - 0.5
        norm = 2 * index / (self.size - 1) - 1  # align_corners=True: -1/+1 = first/last centre
        grid = norm.flip(-1).reshape(1, 1, 1, -1, 3)
        out = F.grid_sample(
            self.grid, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return out.reshape(lead)


def box_grid(box_min_m: NDArray, box_max_m: NDArray, voxel_size_m: float):
    """Origin and shape of a voxel grid covering the box (rounded up to whole voxels)."""
    lo, hi = np.asarray(box_min_m, float), np.asarray(box_max_m, float)
    if lo.shape != (3,) or hi.shape != (3,) or not (hi > lo).all() or voxel_size_m <= 0:
        raise ValueError("Need box_min < box_max (xyz, metres) and a positive voxel size")
    shape = tuple(int(n) for n in np.ceil((hi - lo) / voxel_size_m - 1e-9))
    return lo, shape


def occupancy(
    points: NDArray, origin_m: NDArray, shape, voxel_size_m: float, min_points: int = 1
) -> NDArray:
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    idx = np.floor((points - origin_m) / voxel_size_m).astype(np.int64)
    idx = idx[((idx >= 0) & (idx < np.array(shape))).all(1)]
    counts = np.zeros(shape, np.int64)
    np.add.at(counts, tuple(idx.T), 1)
    return counts >= min_points


def _trilinear(grid: NDArray, u: NDArray) -> tuple[NDArray, NDArray]:
    """Trilinear value and d/du at continuous indices u [n, 3], clamped to the grid."""
    n = np.array(grid.shape)
    inside = (u >= 0) & (u <= n - 1)
    u = np.clip(u, 0, n - 1)
    i0 = np.minimum(np.floor(u).astype(np.int64), n - 2)
    f = u - i0
    value = np.zeros(len(u))
    grad = np.zeros((len(u), 3))
    for corner in product((0, 1), repeat=3):
        c = np.array(corner)
        g = grid[i0[:, 0] + c[0], i0[:, 1] + c[1], i0[:, 2] + c[2]].astype(np.float64)
        w = np.where(c, f, 1 - f)  # per-axis weights [n, 3]
        value += w.prod(1) * g
        for a in range(3):
            others = np.prod(np.delete(w, a, axis=1), axis=1)
            grad[:, a] += (1 if c[a] else -1) * others * g
    return value, np.where(inside, grad, 0.0)


def _env_spheres(spheres, exclude):
    from so101_semantic.collision import load_spheres

    spheres = load_spheres() if spheres is None else spheres
    return {link: s for link, s in spheres.items() if link not in exclude}


def sphere_clearances(
    model, q: dict[str, float], field: DistanceField, spheres=None, exclude=ENV_EXCLUDED_LINKS
) -> tuple[NDArray, list[str]]:
    """Per-sphere clearance (field distance at the centre minus radius, metres) and its link."""
    centres, radii, links = model.collision_spheres(q, _env_spheres(spheres, exclude))
    distance, _ = field.query(centres)
    return distance - radii, links


def clearance(
    model, q: dict[str, float], field: DistanceField, spheres=None, exclude=ENV_EXCLUDED_LINKS
) -> float:
    """Arm clearance at URDF joint radians ``q``: min over spheres of (distance - radius)."""
    return float(sphere_clearances(model, q, field, spheres, exclude)[0].min())


def clearance_jacobian(
    model,
    q: dict[str, float],
    field: DistanceField,
    spheres=None,
    exclude=ENV_EXCLUDED_LINKS,
    eps: float = 1e-4,
) -> tuple[NDArray, NDArray, list[str]]:
    """Per-sphere clearance [n], d clearance / d q [n, len(JOINT_NAMES)] (per radian) and links.

    Chain rule: field gradient at each centre times the centre's motion per joint (central
    differences of FK). For a linearised CBF on clearance; the field gradient is piecewise
    constant within a voxel.
    """
    env = _env_spheres(spheres, exclude)
    centres, radii, links = model.collision_spheres(q, env)
    distance, grad = field.query(centres)
    jac = np.zeros((len(radii), len(JOINT_NAMES)))
    for j, name in enumerate(JOINT_NAMES):
        plus, minus = dict(q), dict(q)
        plus[name] += eps
        minus[name] -= eps
        dc = (model.collision_spheres(plus, env)[0] - model.collision_spheres(minus, env)[0]) / (
            2 * eps
        )
        jac[:, j] = np.einsum("ni,ni->n", grad, dc)
    return distance - radii, jac, links
