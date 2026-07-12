# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices

np.random.seed(0)


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
def test_knn_index(device):
    dtype = o3c.float32

    t = o3c.Tensor.zeros((10, 3), dtype, device=device)
    nns = o3c.nns.NearestNeighborSearch(t)
    assert nns.knn_index()
    assert nns.fixed_radius_index(0.1)
    assert nns.hybrid_index(0.1)

    # Multi radii search is only supported on CPU.
    if device.get_type() == o3c.Device.DeviceType.CPU:
        assert nns.multi_radius_index()


@pytest.mark.parametrize("device",
                         list_devices(enable_sycl=True, also_sycl_cpu=False))
def test_knn_search(device):
    dtype = o3c.float32

    dataset_points = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=dtype,
        device=device)
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.knn_index()

    # Single query point.
    query_points = o3c.Tensor([[0.064705, 0.043921, 0.087843]],
                              dtype=dtype,
                              device=device)
    indices, distances = nns.knn_search(query_points, 3)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([[1, 4, 9]], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([[0.00626358, 0.00747938, 0.0108912]],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)

    # Multiple query points.
    query_points = o3c.Tensor(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=dtype,
        device=device)
    indices, distances = nns.knn_search(query_points, 3)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([[1, 4, 9], [1, 4, 9]], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([[0.00626358, 0.00747938, 0.0108912],
                                         [0.00626358, 0.00747938, 0.0108912]],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)

    # Test knn search with large k (>2048)
    dataset_points_np = np.random.randn(5000, 3).astype(np.float32) * 100
    dataset_points = o3c.Tensor(dataset_points_np, dtype=dtype, device=device)
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.knn_index()

    query_point_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    query_point = o3c.Tensor(query_point_np, dtype=dtype, device=device)

    knn = 3100
    indices, distances = nns.knn_search(query_point, knn)

    indices_np = indices.cpu().numpy()
    distances_np = distances.cpu().numpy()

    # Compute ground truth using brute force
    gt_distances = np.sum((dataset_points_np - query_point_np[0])**2, axis=1)
    gt_sorted_indices = np.argsort(gt_distances)[:knn]
    gt_sorted_distances = gt_distances[gt_sorted_indices]

    # Check for duplicate indices
    np.testing.assert_equal(len(set(indices_np[0])), knn)
    # verify idx values are in valid range
    np.testing.assert_array_less(indices_np[0], dataset_points.shape[0])
    np.testing.assert_array_less(-1, indices_np[0])
    # Verify distances match ground truth
    np.testing.assert_allclose(distances_np[0],
                               gt_sorted_distances,
                               atol=1e-4,
                               rtol=1e-5)


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_fixed_radius_search(device, dtype):
    dataset_points = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=dtype,
        device=device)
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.fixed_radius_index(0.1)

    # Single query point.
    query_points = o3c.Tensor([[0.064705, 0.043921, 0.087843]],
                              dtype=dtype,
                              device=device)
    indices, distances, neighbors_row_splits = nns.fixed_radius_search(
        query_points, 0.1)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([1, 4], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([0.00626358, 0.00747938],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)
    np.testing.assert_equal(neighbors_row_splits.cpu().numpy(),
                            np.array([0, 2], dtype=np.int64))

    # Multiple query points.
    query_points = o3c.Tensor(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=dtype,
        device=device)
    indices, distances, neighbors_row_splits = nns.fixed_radius_search(
        query_points, 0.1)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([1, 4, 1, 4], dtype=np.int64))
    np.testing.assert_allclose(
        distances.cpu().numpy(),
        np.array([0.00626358, 0.00747938, 0.00626358, 0.00747938],
                 dtype=np.float64),
        rtol=1e-5,
        atol=0)
    np.testing.assert_equal(neighbors_row_splits.cpu().numpy(),
                            np.array([0, 2, 4], dtype=np.int64))


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_hybrid_search_random(dtype):
    if o3c.cuda.device_count() > 0:
        dataset_size, query_size = 1000, 100
        radius, k = 0.1, 10

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        dataset_points_cuda = dataset_points.cuda()

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_cuda = o3c.nns.NearestNeighborSearch(dataset_points_cuda)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_cuda = query_points.cuda()

            nns.hybrid_index(radius)
            indices, distances, counts = nns.hybrid_search(
                query_points, radius, k)

            nns_cuda.hybrid_index(radius)
            indices_cuda, distances_cuda, counts_cuda = nns_cuda.hybrid_search(
                query_points_cuda, radius, k)

            np.testing.assert_allclose(distances.numpy(),
                                       distances_cuda.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_cuda.cpu().numpy())
            np.testing.assert_equal(counts.numpy(), counts_cuda.cpu().numpy())

    # SYCL: compare SYCL results against CPU reference. Also exercises the
    # CPU-fallback SYCL device (len == 1) now that fixed-radius/hybrid search
    # support SYCL CPU via the uniform-grid algorithm.
    if len(o3c.sycl.get_available_devices()) >= 1:
        dataset_size, query_size = 1000, 100
        radius, k = 0.1, 10

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        sycl_device = o3c.Device("SYCL:0")
        dataset_points_sycl = dataset_points.to(sycl_device)

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_sycl = o3c.nns.NearestNeighborSearch(dataset_points_sycl)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_sycl = query_points.to(sycl_device)

            nns.hybrid_index(radius)
            indices, distances, counts = nns.hybrid_search(
                query_points, radius, k)

            nns_sycl.hybrid_index(radius)
            indices_sycl, distances_sycl, counts_sycl = nns_sycl.hybrid_search(
                query_points_sycl, radius, k)

            np.testing.assert_allclose(distances.numpy(),
                                       distances_sycl.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_sycl.cpu().numpy())
            np.testing.assert_equal(counts.numpy(), counts_sycl.cpu().numpy())


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_fixed_radius_search_random(dtype):
    if o3c.cuda.device_count() > 0:
        dataset_size, query_size = 1000, 100
        radius = 0.1

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        dataset_points_cuda = dataset_points.cuda()

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_cuda = o3c.nns.NearestNeighborSearch(dataset_points_cuda)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_cuda = query_points.cuda()

            nns.fixed_radius_index(radius)
            indices, distances, neighbors_row_splits = nns.fixed_radius_search(
                query_points, radius)

            nns_cuda.fixed_radius_index(radius)
            indices_cuda, distances_cuda, neighbors_row_splits_cuda = nns_cuda.fixed_radius_search(
                query_points_cuda, radius)

            np.testing.assert_equal(neighbors_row_splits.numpy(),
                                    neighbors_row_splits_cuda.cpu().numpy())
            np.testing.assert_allclose(distances.numpy(),
                                       distances_cuda.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_cuda.cpu().numpy())

    # SYCL: compare SYCL results against CPU reference. Also exercises the
    # CPU-fallback SYCL device (len == 1) now that fixed-radius/hybrid search
    # support SYCL CPU via the uniform-grid algorithm.
    if len(o3c.sycl.get_available_devices()) >= 1:
        dataset_size, query_size = 1000, 100
        radius = 0.1

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        sycl_device = o3c.Device("SYCL:0")
        dataset_points_sycl = dataset_points.to(sycl_device)

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_sycl = o3c.nns.NearestNeighborSearch(dataset_points_sycl)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_sycl = query_points.to(sycl_device)

            nns.fixed_radius_index(radius)
            indices, distances, neighbors_row_splits = nns.fixed_radius_search(
                query_points, radius)

            nns_sycl.fixed_radius_index(radius)
            indices_sycl, distances_sycl, \
                neighbors_row_splits_sycl = nns_sycl.fixed_radius_search(
                    query_points_sycl, radius)

            np.testing.assert_equal(neighbors_row_splits.numpy(),
                                    neighbors_row_splits_sycl.cpu().numpy())
            np.testing.assert_allclose(distances.numpy(),
                                       distances_sycl.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_sycl.cpu().numpy())


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_knn_search_sycl_matches_cpu(dtype):
    """KNN on SYCL:0 should match CPU for the canonical grid fixture."""
    if len(o3c.sycl.get_available_devices()) <= 1:
        pytest.skip("SYCL GPU not available.")

    dataset_points = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=dtype,
        device=o3c.Device("CPU:0"))
    query_points = o3c.Tensor([[0.064705, 0.043921, 0.087843]],
                              dtype=dtype,
                              device=o3c.Device("CPU:0"))

    nns_cpu = o3c.nns.NearestNeighborSearch(dataset_points)
    nns_cpu.knn_index()
    indices_cpu, distances_cpu = nns_cpu.knn_search(query_points, 3)

    sycl = o3c.Device("SYCL:0")
    nns_sycl = o3c.nns.NearestNeighborSearch(dataset_points.to(sycl))
    nns_sycl.knn_index()
    indices_sycl, distances_sycl = nns_sycl.knn_search(query_points.to(sycl), 3)

    np.testing.assert_equal(indices_cpu.numpy(), indices_sycl.cpu().numpy())
    np.testing.assert_allclose(distances_cpu.numpy(),
                               distances_sycl.cpu().numpy(),
                               rtol=1e-5,
                               atol=0)


# ── SYCL correctness regression tests ────────────────────────────────────────
# These tests target specific bug fixes in the SYCL NNS implementation.
# They are skipped when no SYCL GPU is available (i.e. only the CPU fallback
# device is listed).  Each test documents the bug it covers.


def _sycl_skip_if_no_gpu():
    """Skip the calling test when no SYCL GPU is present."""
    if len(o3c.sycl.get_available_devices()) <= 1:
        pytest.skip("SYCL GPU not available.")


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_knn_search_sycl_coincident_c1(dtype):
    """C1: a query coincident with a dataset point must produce distance >= 0.

    Floating-point rounding in −2*q*p + |p|² can yield a tiny negative value
    when q == p.  The fix clamps partial distances to zero before comparison.
    """
    _sycl_skip_if_no_gpu()
    sycl = o3c.Device("SYCL:0")

    # Place a query exactly on one of the dataset points.
    dataset = o3c.Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                         dtype=dtype,
                         device=sycl)
    query = o3c.Tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=sycl)

    nns = o3c.nns.NearestNeighborSearch(dataset)
    nns.knn_index()
    _, distances = nns.knn_search(query, 1)

    dist = distances.cpu().numpy()[0, 0]
    assert dist >= 0.0, f"C1 violated: distance {dist} < 0 for coincident query"


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_knn_search_sycl_equidistant_tiebreak_c4(dtype):
    """C4: equidistant neighbors must be returned with consistent tie-breaking.

    When two dataset points are the same distance from a query the SYCL result
    must agree with the CPU result (smaller global index wins).
    """
    _sycl_skip_if_no_gpu()
    sycl = o3c.Device("SYCL:0")

    # Three points, last two equidistant from the query.
    dataset_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                          dtype=np.float32)
    query_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    dataset_cpu = o3c.Tensor(dataset_np,
                             dtype=dtype,
                             device=o3c.Device("CPU:0"))
    query_cpu = o3c.Tensor(query_np, dtype=dtype, device=o3c.Device("CPU:0"))

    nns_cpu = o3c.nns.NearestNeighborSearch(dataset_cpu)
    nns_cpu.knn_index()
    indices_cpu, distances_cpu = nns_cpu.knn_search(query_cpu, 3)

    dataset_sycl = dataset_cpu.to(sycl)
    query_sycl = query_cpu.to(sycl)

    nns_sycl = o3c.nns.NearestNeighborSearch(dataset_sycl)
    nns_sycl.knn_index()
    indices_sycl, distances_sycl = nns_sycl.knn_search(query_sycl, 3)

    np.testing.assert_equal(indices_cpu.numpy(), indices_sycl.cpu().numpy())
    np.testing.assert_allclose(distances_cpu.numpy(),
                               distances_sycl.cpu().numpy(),
                               rtol=1e-5,
                               atol=1e-6)


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_knn_search_sycl_k_exceeds_num_points_c5(dtype):
    """C5: requesting knn > num_points must not crash or return garbage.

    The SYCL driver must clamp the effective k per batch so that index and
    distance buffers are not read out-of-bounds.
    """
    _sycl_skip_if_no_gpu()
    sycl = o3c.Device("SYCL:0")

    dataset_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                          dtype=np.float32)
    query_np = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)

    dataset = o3c.Tensor(dataset_np, dtype=dtype, device=sycl)
    query = o3c.Tensor(query_np, dtype=dtype, device=sycl)

    nns = o3c.nns.NearestNeighborSearch(dataset)
    nns.knn_index()
    # Request more neighbors than exist in the dataset.
    indices, distances = nns.knn_search(query, 10)

    idx_np = indices.cpu().numpy()[0]
    dist_np = distances.cpu().numpy()[0]

    # All returned indices must be valid.
    assert np.all(idx_np >= 0) and np.all(
        idx_np < 3), f"C5: out-of-range indices {idx_np}"
    # All returned distances must be non-negative.
    assert np.all(dist_np >= 0), f"C5: negative distances {dist_np}"
    # The closest point (index 0, distance 0.01) must be first.
    assert idx_np[0] == 0, f"C5: wrong nearest index {idx_np[0]}, expected 0"
    np.testing.assert_allclose(dist_np[0], 0.01, rtol=1e-4)
