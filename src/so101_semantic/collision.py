"""Collision spheres for the SO-101 links, fitted offline to the vendored collision meshes.

Spheres are stored per link in the link frame (metres) in ``configs/so101_collision_spheres.yaml``
and placed with FK by ``SO101Model.collision_spheres``. Fitting needs the meshes
(``scripts/fetch_so101_meshes.py``); loading and placing the stored spheres does not.
"""

from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
SPHERES_PATH = ROOT / "configs" / "so101_collision_spheres.yaml"


def load_spheres(path: Path = SPHERES_PATH) -> dict[str, tuple[NDArray, NDArray]]:
    """Link name -> (centres [k, 3], radii [k]) in the link frame."""
    links = yaml.safe_load(path.read_text())["links"]
    return {
        name: (
            np.array([s["center"] for s in spheres], dtype=float),
            np.array([s["radius"] for s in spheres], dtype=float),
        )
        for name, spheres in links.items()
    }


def link_mesh(urdf, link: str, model_dir: Path):
    """The link's collision meshes merged into one trimesh in the link frame, or None."""
    import trimesh

    parts = []
    for collision in urdf.link_map[link].collisions:
        mesh = trimesh.load(str(model_dir / collision.geometry.mesh.filename), force="mesh")
        if collision.origin is not None:
            mesh.apply_transform(collision.origin)
        parts.append(mesh)
    return trimesh.util.concatenate(parts) if parts else None


def surface_points(mesh, n_samples: int, seed: int) -> NDArray:
    """Vertices plus uniform surface samples, so sphere coverage is checked between vertices too."""
    import trimesh

    samples, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    return np.vstack([mesh.vertices, samples])


def _enclosing_ball(points: NDArray, iterations: int = 200) -> tuple[NDArray, float]:
    """Approximate minimum enclosing ball (Badoiu-Clarkson); the radius always covers ``points``."""
    centre = points.mean(axis=0)
    for i in range(1, iterations + 1):
        far = points[np.argmax(np.linalg.norm(points - centre, axis=1))]
        centre = centre + (far - centre) / (i + 1)
    return centre, float(np.linalg.norm(points - centre, axis=1).max())


def fit_spheres(
    points: NDArray, k: int, seed: int = 0, rounds: int = 30, subsample: int = 6000
) -> tuple[NDArray, NDArray]:
    """Cover ``points`` with ``k`` spheres of small total volume; every point ends up inside one.

    Centres are optimised on a random subsample; final radii are sized over all points.
    """
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    fit = points[rng.choice(len(points), min(subsample, len(points)), replace=False)]
    centres, labels = kmeans2(fit, k, minit="++", seed=seed)
    for _ in range(rounds):
        balls = [_enclosing_ball(fit[labels == i]) for i in range(k) if np.any(labels == i)]
        centres = np.array([c for c, _ in balls])
        radii = np.array([r for _, r in balls])
        # Each point joins the sphere it overflows least, which lets small spheres stay small.
        overflow = np.linalg.norm(fit[:, None] - centres[None], axis=2) - radii[None]
        new_labels = np.argmin(overflow, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    overflow = np.linalg.norm(points[:, None] - centres[None], axis=2) - radii[None]
    labels = np.argmin(overflow, axis=1)
    radii = np.array(
        [
            np.linalg.norm(points[labels == i] - c, axis=1).max(initial=0.0)
            for i, c in enumerate(centres)
        ]
    )
    keep = radii > 0
    return centres[keep], radii[keep]


def bulge(mesh, centres: NDArray, radii: NDArray, n: int = 4000, seed: int = 0) -> NDArray:
    """Distances (m) from the outer surface of the sphere union to the link's convex hull.

    How much free space the spheres claim beyond the link itself: clearance is
    underestimated by up to this much in the worst direction.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surface = (centres[:, None] + radii[:, None, None] * dirs[None]).reshape(-1, 3)
    gaps = np.linalg.norm(surface[:, None] - centres[None], axis=2) - radii[None]
    outer = surface[gaps.min(axis=1) >= -1e-9]
    hull = mesh.convex_hull
    return np.maximum(-hull.nearest.signed_distance(outer), 0.0)


def union_volume(centres: NDArray, radii: NDArray, n: int = 200_000, seed: int = 0) -> float:
    """Monte Carlo volume of a union of spheres (m^3)."""
    rng = np.random.default_rng(seed)
    lo = (centres - radii[:, None]).min(axis=0)
    hi = (centres + radii[:, None]).max(axis=0)
    samples = rng.uniform(lo, hi, size=(n, 3))
    inside = np.zeros(n, dtype=bool)
    for c, r in zip(centres, radii, strict=True):
        inside |= np.linalg.norm(samples - c, axis=1) <= r
    return float(inside.mean() * np.prod(hi - lo))


def max_protrusion(points: NDArray, centres: NDArray, radii: NDArray) -> float:
    """Largest distance (m) by which any point lies outside every sphere; <= 0 means covered."""
    gaps = np.linalg.norm(points[:, None] - centres[None], axis=2) - radii[None]
    return float(gaps.min(axis=1).max())


def load_ignored_pairs(path: Path = SPHERES_PATH) -> set[frozenset[str]]:
    """Link pairs excluded from self-collision checks (adjacent, or proven unable to touch)."""
    record = yaml.safe_load(path.read_text())
    return {frozenset(entry["links"]) for entry in record["self_collision"]["ignore"]}


def self_collision_pairs(links, ignored: set[frozenset[str]]) -> list[tuple[str, str]]:
    links = list(links)
    return [
        (a, b)
        for i, a in enumerate(links)
        for b in links[i + 1 :]
        if frozenset((a, b)) not in ignored
    ]


def joints_between(urdf, a: str, b: str) -> list[str] | None:
    """Movable joints between links ``a`` and ``b``; None if neither is the other's ancestor."""
    parent = {j.child: j for j in urdf.robot.joints}

    def up(link: str, stop: str) -> list[str] | None:
        joints = []
        while link != stop:
            if link not in parent:
                return None
            joint = parent[link]
            if joint.type != "fixed":
                joints.append(joint.name)
            link = joint.parent
        return joints

    found = up(b, a)
    return found if found is not None else up(a, b)


def sample_spacing(points: NDArray) -> float:
    """Largest nearest-neighbour gap in a point sample: the surface lies within half of it."""
    from scipy.spatial import cKDTree

    return float(cKDTree(points).query(points, k=2)[0][:, 1].max())


def even_surface_points(mesh, n_samples: int, seed: int) -> NDArray:
    """Roughly evenly spaced surface samples plus vertices, for tight sampling allowances."""
    import trimesh

    samples, _ = trimesh.sample.sample_surface_even(mesh, n_samples, seed=seed)
    return np.vstack([mesh.vertices, samples])


def _sweep_poses(model, a: str, b: str, joints: list[str], grids: list[NDArray]):
    """Yield (b-in-a transform, per-joint axis point and direction in a's frame) over the grid."""
    import itertools

    q0 = dict.fromkeys(model.urdf.actuated_joint_names, 0.0)
    for values in itertools.product(*grids):
        model.urdf.update_cfg({**q0, **dict(zip(joints, values, strict=True))})
        inv_a = np.linalg.inv(np.array(model.urdf.get_transform(a, model.base_frame)))
        rel = inv_a @ np.array(model.urdf.get_transform(b, model.base_frame))
        axes = []
        for j in joints:
            joint = model.urdf.joint_map[j]
            t = inv_a @ np.array(model.urdf.get_transform(joint.child, model.base_frame))
            axes.append((t[:3, 3], t[:3, :3] @ np.asarray(joint.axis, dtype=float)))
        yield rel, axes


def _reach(centres: NDArray, radii: NDArray, origin: NDArray, axis: NDArray) -> float:
    """Largest distance from a joint axis to any point inside the given spheres."""
    axis = axis / np.linalg.norm(axis)
    return float((np.linalg.norm(np.cross(centres - origin, axis), axis=1) + radii).max())


def pair_distance_bound(
    model,
    a: str,
    b: str,
    points: dict[str, NDArray],
    spheres: dict[str, tuple[NDArray, NDArray]],
    motion_per_joint: float = 0.002,
    max_steps: int = 400,
) -> tuple[float, list[str]]:
    """Lower bound (m) on the mesh distance between links ``a`` and ``b`` over their joint ranges.

    ``a`` must be the ancestor of ``b``; ``points`` are surface samples in each link frame.
    Their relative pose depends only on the joints between them, which are swept on a full grid
    (others held at zero), with each joint's step chosen so ``b`` moves at most
    ``motion_per_joint`` between grid points. At each grid point the distance is bounded below by
    the sphere distance (spheres cover the meshes) or, where spheres come close, by the sampled
    mesh distance minus the sampling gaps. Subtracting the motion between grid points makes the
    bound hold over the whole range, so a positive value means the links cannot touch.
    """
    from scipy.spatial import cKDTree

    joints = joints_between(model.urdf, a, b)
    (ca, ra), (cb, rb) = spheres[a], spheres[b]
    ranges = [model.limits(j) for j in joints]

    # Coarse pass: how far b's spheres can sit from each joint axis.
    reach = np.zeros(len(joints))
    coarse = [np.linspace(lo, hi, 12) for lo, hi in ranges]
    for rel, axes in _sweep_poses(model, a, b, joints, coarse):
        cb_in_a = cb @ rel[:3, :3].T + rel[:3, 3]
        reach = np.maximum(reach, [_reach(cb_in_a, rb, o, d) for o, d in axes])
    reach *= 1.05  # the coarse grid can miss the exact maximum
    steps = [
        int(min(max_steps, np.ceil((hi - lo) * r / (2 * motion_per_joint)) + 1))
        for (lo, hi), r in zip(ranges, reach, strict=True)
    ]
    grids = [np.linspace(lo, hi, n) for (lo, hi), n in zip(ranges, steps, strict=True)]
    motion = sum((g[1] - g[0]) / 2 * r for g, r in zip(grids, reach, strict=True))

    tree = cKDTree(points[a])
    sampling = sample_spacing(points[a]) + sample_spacing(points[b])
    near = motion + sampling + 0.01  # beyond this, the sphere bound already decides
    lowest = np.inf
    for rel, _ in _sweep_poses(model, a, b, joints, grids):
        cb_in_a = cb @ rel[:3, :3].T + rel[:3, 3]
        bound = (np.linalg.norm(ca[:, None] - cb_in_a[None], axis=2) - ra[:, None] - rb[None]).min()
        if bound < near:
            pb = points[b] @ rel[:3, :3].T + rel[:3, 3]
            gap = (np.linalg.norm(pb[:, None] - ca[None], axis=2) - ra[None]).min(axis=1)
            close = pb[gap < near]
            nearest = (
                tree.query(close, k=1, distance_upper_bound=near)[0].min() if len(close) else near
            )
            bound = max(bound, min(nearest, near) - sampling)
        lowest = min(lowest, bound)
    return float(lowest - motion), joints
