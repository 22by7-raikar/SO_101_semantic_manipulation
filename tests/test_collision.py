import hashlib

import numpy as np
import pytest
import yaml

from so101_semantic.collision import (
    SPHERES_PATH,
    bulge,
    fit_spheres,
    link_mesh,
    load_spheres,
    max_protrusion,
    surface_points,
)
from so101_semantic.kinematics import JOINT_NAMES, URDF_PATH, SO101Model

MESHES_PRESENT = (URDF_PATH.parent / "assets" / "sts3215_03a_v1.stl").exists()
needs_meshes = pytest.mark.skipif(
    not MESHES_PRESENT, reason="meshes not fetched (scripts/fetch_so101_meshes.py)"
)


@pytest.fixture(scope="module")
def model():
    return SO101Model()


@pytest.fixture(scope="module")
def spheres():
    return load_spheres()


def home():
    return dict.fromkeys(JOINT_NAMES, 0.0)


def test_spheres_match_vendored_urdf():
    record = yaml.safe_load(SPHERES_PATH.read_text())
    assert record["urdf_sha256"] == hashlib.sha256(URDF_PATH.read_bytes()).hexdigest(), (
        "URDF changed; rerun scripts/fit_collision_spheres.py"
    )


def test_every_link_with_geometry_has_spheres(model, spheres):
    with_geometry = {name for name, link in model.urdf.link_map.items() if link.collisions}
    assert set(spheres) == with_geometry
    for centres, radii in spheres.values():
        assert centres.shape == (len(radii), 3)
        assert 3 <= len(radii) <= 5
        assert np.all((radii > 0.005) & (radii < 0.06))


def test_spheres_follow_fk(model, spheres):
    centres, radii, links = model.collision_spheres(home(), spheres)
    assert len(centres) == len(radii) == len(links) == sum(len(r) for _, r in spheres.values())
    t = model.fk(home(), "gripper_link")
    local = spheres["gripper_link"][0]
    placed = centres[[i for i, link in enumerate(links) if link == "gripper_link"]]
    np.testing.assert_allclose(placed, local @ t[:3, :3].T + t[:3, 3], atol=1e-12)


def test_spheres_move_rigidly_with_their_link(model, spheres):
    q = {**home(), "shoulder_pan": 0.7, "elbow_flex": -0.9, "wrist_roll": 1.2}
    a, _, links = model.collision_spheres(home(), spheres)
    b, _, _ = model.collision_spheres(q, spheres)
    links = np.array(links)
    base = links == "base_link"
    np.testing.assert_allclose(a[base], b[base])
    for link in set(links) - {"base_link"}:
        sel = links == link
        da = np.linalg.norm(a[sel][:, None] - a[sel][None], axis=2)
        db = np.linalg.norm(b[sel][:, None] - b[sel][None], axis=2)
        np.testing.assert_allclose(da, db, atol=1e-9)
        assert not np.allclose(a[sel], b[sel], atol=1e-3), f"{link} did not move"


def test_jaw_spheres_follow_gripper_joint(model, spheres):
    lower, upper = model.limits("gripper")
    a, _, links = model.collision_spheres({**home(), "gripper": lower}, spheres)
    b, _, _ = model.collision_spheres({**home(), "gripper": upper}, spheres)
    jaw = np.array(links) == "moving_jaw_so101_v1_link"
    assert np.linalg.norm(a[jaw] - b[jaw], axis=1).max() > 0.02
    np.testing.assert_allclose(a[~jaw], b[~jaw])


def test_fit_covers_points_and_protrusion_measure():
    rng = np.random.default_rng(0)
    points = rng.uniform([-0.1, -0.01, -0.01], [0.1, 0.01, 0.01], size=(3000, 3))  # a bar
    centres, radii = fit_spheres(points, 4)
    assert len(radii) == 4
    assert max_protrusion(points, centres, radii) <= 1e-12
    assert radii.max() < 0.05  # spread along the bar, not one big sphere
    assert max_protrusion(np.array([[1.0, 0.0, 0.0]]), centres, radii) > 0.8


@needs_meshes
def test_stored_spheres_cover_each_link_mesh(model, spheres):
    for link, (centres, radii) in spheres.items():
        mesh = link_mesh(model.urdf, link, URDF_PATH.parent)
        # A fresh sample, independent of the one used for fitting.
        points = surface_points(mesh, 20000, seed=123)
        assert max_protrusion(points, centres, radii) <= 0.0, link


@needs_meshes
def test_stored_spheres_stay_reasonably_tight(model, spheres):
    # Regression bound on how much free space the spheres claim beyond the convex hull.
    for link, (centres, radii) in spheres.items():
        b = bulge(link_mesh(model.urdf, link, URDF_PATH.parent), centres, radii, n=1000)
        assert np.mean(b) < 0.015, link
        assert np.percentile(b, 95) < 0.035, link


# Self-collision


def test_adjacent_pairs_are_ignored_and_others_justified(model):
    ignored = yaml.safe_load(SPHERES_PATH.read_text())["self_collision"]["ignore"]
    reasons = {frozenset(e["links"]): e["reason"] for e in ignored}
    for joint in model.urdf.robot.joints:
        if model.urdf.link_map[joint.child].collisions:
            assert reasons.get(frozenset((joint.parent, joint.child))) == "adjacent"
    for pair, reason in reasons.items():
        assert reason == "adjacent" or reason.startswith("meshes stay >= "), pair


def test_checked_pairs_cover_every_non_ignored_pair(model, spheres):
    from so101_semantic.collision import load_ignored_pairs, self_collision_pairs

    pairs = self_collision_pairs(spheres, load_ignored_pairs())
    assert len(pairs) + len(load_ignored_pairs()) == len(spheres) * (len(spheres) - 1) // 2
    assert set(model.self_collision_distances(home())) == set(pairs)


def test_home_is_free(model):
    assert model.self_collisions(home()) == []


def test_folding_the_elbow_fully_collides(model):
    lower, upper = model.limits("elbow_flex")
    q = {**home(), "shoulder_lift": model.limits("shoulder_lift")[0], "elbow_flex": upper}
    assert model.self_collisions(q)


def _samples(model, links, n):
    import trimesh

    return {
        link: trimesh.sample.sample_surface(
            link_mesh(model.urdf, link, URDF_PATH.parent), n, seed=5
        )[0]
        for link in links
    }


def _mesh_distance(model, q, trees, samples, a, b):
    """Sampled mesh distance between links a and b; never below the true distance."""
    rel = np.linalg.inv(model.fk(q, a)) @ model.fk(q, b)
    return trees[a].query(samples[b] @ rel[:3, :3].T + rel[:3, 3], k=1)[0].min()


@needs_meshes
def test_sphere_distance_never_exceeds_mesh_distance(model, spheres):
    """Spheres cover the meshes, so a real contact can never be missed."""
    from scipy.spatial import cKDTree

    samples = _samples(model, spheres, 1500)
    trees = {link: cKDTree(points) for link, points in samples.items()}
    rng = np.random.default_rng(2)
    for _ in range(25):
        q = {name: rng.uniform(*model.limits(name)) for name in JOINT_NAMES}
        for (a, b), sphere_d in model.self_collision_distances(q).items():
            assert sphere_d <= _mesh_distance(model, q, trees, samples, a, b) + 1e-9, (a, b, q)


@needs_meshes
def test_pairs_proven_apart_stay_apart(model):
    from scipy.spatial import cKDTree

    ignored = yaml.safe_load(SPHERES_PATH.read_text())["self_collision"]["ignore"]
    proven = [e for e in ignored if e["reason"] != "adjacent"]
    samples = _samples(model, {link for e in proven for link in e["links"]}, 3000)
    trees = {link: cKDTree(points) for link, points in samples.items()}
    rng = np.random.default_rng(3)
    for entry in proven:
        a, b = entry["links"]
        bound_m = float(entry["reason"].split(">= ")[1].split(" mm")[0]) / 1000
        for _ in range(40):
            q = {name: rng.uniform(*model.limits(name)) for name in JOINT_NAMES}
            # Sampled points can only overestimate the true distance, which is >= the bound.
            assert _mesh_distance(model, q, trees, samples, a, b) >= bound_m > 0, (a, b, q)
