from __future__ import annotations

import numpy as np
import pytest
from vedo import Mesh as VedoMesh
from vedo import Points as VedoPoints

from mojovedo import Mesh, Points


TRI_POINTS = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
TRI_FACES = np.array([[0, 1, 2]])


def test_distance_to_surface_matches_vedo():
    query = np.array([[0.2, 0.2, 1], [2, 0, 0], [-0.5, -0.5, 0]])
    ours = Points(query).distance_to(Mesh([TRI_POINTS, TRI_FACES]))
    theirs = VedoPoints(query).distance_to(VedoMesh([TRI_POINTS, TRI_FACES]))
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_point_distances_chamfer_and_hausdorff_match_vedo():
    a = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    b = np.array([[1.0, 0, 0], [0, 1, 0]])
    ours_a, ours_b = Points(a), Points(b)
    vedo_a, vedo_b = VedoPoints(a), VedoPoints(b)
    assert ours_a.chamfer_distance(ours_b) == pytest.approx(vedo_a.chamfer_distance(vedo_b))
    assert ours_a.hausdorff_distance(ours_b) == pytest.approx(vedo_a.hausdorff_distance(vedo_b))


def test_point_distance_simd_tail_matches_numpy():
    query = np.array([[.2, -.3, .4], [2.1, .2, -.1], [-.5, .8, 1.2]])
    target = np.arange(39, dtype=np.float64).reshape(13, 3) / 7.0
    got = Points(query).distance_to(Points(target))
    expected = np.sqrt(((query[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_point_distance_parallel_threshold_matches_numpy():
    rng = np.random.default_rng(19)
    query = rng.normal(size=(1_001, 3))
    target = rng.normal(size=(1_000, 3))
    got = Points(query).distance_to(Points(target))
    expected = np.array([np.sqrt(((target - point) ** 2).sum(axis=1)).min() for point in query])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_chamfer_parallel_combined_schedule_with_simd_tails():
    rng = np.random.default_rng(23)
    a = rng.normal(size=(513, 3))
    b = rng.normal(size=(511, 3))
    got = Points(a).chamfer_distance(Points(b))
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    expected = (distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_ffi_input_validation_prevents_lossy_casts_and_empty_chamfer():
    with pytest.raises(TypeError, match="integer dtype"):
        Mesh([TRI_POINTS, np.array([[0.0, 1.0, 2.0]])])
    with pytest.raises(ValueError, match="exactly representable"):
        Points(np.array([[2**53 + 1, 0, 0]], dtype=np.int64))
    with pytest.raises(ValueError, match="non-empty"):
        Points(np.empty((0, 3))).chamfer_distance(Points([[0, 0, 0]]))


def test_densify_rejects_capacity_smaller_than_input_before_ffi_call():
    with pytest.raises(ValueError, match="at least"):
        Points([[0, 0, 0], [1, 0, 0]]).densify(nmax=1)


@pytest.mark.parametrize("metric", [0, 2, 6, 8, 28])
def test_quality_metrics_match_vedo(metric):
    ours = Mesh([TRI_POINTS, TRI_FACES]).compute_quality(metric).celldata["Quality"]
    theirs = VedoMesh([TRI_POINTS, TRI_FACES]).compute_quality(metric).celldata["Quality"]
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_cell_size_matches_vedo():
    ours = Mesh([TRI_POINTS, TRI_FACES]).compute_cell_size()
    theirs = VedoMesh([TRI_POINTS, TRI_FACES]).compute_cell_size()
    for name in ("Area", "Volume", "Length"):
        np.testing.assert_allclose(ours.celldata[name], theirs.celldata[name])


@pytest.mark.parametrize("invert", [False, True])
def test_plane_clip_preserves_the_same_area_as_vedo(invert):
    ours = Mesh([TRI_POINTS, TRI_FACES]).cut_with_plane((0.25, 0, 0), (1, 0, 0), invert)
    theirs = VedoMesh([TRI_POINTS, TRI_FACES]).cut_with_plane((0.25, 0, 0), (1, 0, 0), invert)
    assert ours.area() == pytest.approx(theirs.area(), abs=1e-12)
    assert ours.ncells == theirs.ncells


def test_densify_matches_upstream_invariant_and_keeps_input_points():
    source = np.array([[0.0, 0, 0], [1, 0, 0]])
    ours = Points(source).densify(target_distance=0.5, nclosest=1)
    theirs = VedoPoints(source).densify(target_distance=0.5, nclosest=1)
    for point in source:
        assert np.any(np.all(np.isclose(ours.coordinates, point), axis=1))
        assert np.any(np.all(np.isclose(theirs.coordinates, point), axis=1))
    midpoint = np.array([0.5, 0, 0])
    assert np.any(np.all(np.isclose(ours.coordinates, midpoint), axis=1))
    assert np.any(np.all(np.isclose(theirs.coordinates, midpoint), axis=1))


def test_boolean_is_exact_for_disjoint_and_contained_surfaces():
    tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    outer = Mesh([tetra, faces])
    far = Mesh([tetra + [5, 0, 0], faces])
    assert outer.boolean("plus", far).ncells == 8
    assert outer.boolean("intersect", far).ncells == 0
    assert outer.boolean("minus", far).ncells == 4

    large = Mesh([tetra * 4 - 1, faces])
    small = Mesh([tetra * .25, faces])
    assert large.boolean("plus", small).volume() == pytest.approx(large.volume())
    assert large.boolean("intersect", small).volume() == pytest.approx(small.volume())
    assert large.boolean("minus", small).volume() == pytest.approx(large.volume() - small.volume())


def test_boolean_rejects_open_and_intersecting_surfaces():
    with pytest.raises(ValueError, match="closed"):
        Mesh([TRI_POINTS, TRI_FACES]).boolean("plus", Mesh([TRI_POINTS + 2, TRI_FACES]))
    tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    with pytest.raises(NotImplementedError, match="face splitting"):
        Mesh([tetra, faces]).boolean("plus", Mesh([tetra + .2, faces]))


def test_signed_distance_for_closed_tetrahedron():
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    got = Points([[.1, .1, .1], [2, 2, 2]]).distance_to(Mesh([points, faces]), signed=True)
    assert got[0] < 0 < got[1]
