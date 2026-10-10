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
