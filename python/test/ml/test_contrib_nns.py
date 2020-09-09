import open3d.core as o3c
import numpy as np
import pytest
from open3d.ml.contrib import knn_search, radius_search


def test_knn_search():
    query_points = np.array(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=np.float32)
    dataset_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=np.float32)
    knn = 3

    indices = knn_search(o3c.Tensor.from_numpy(query_points),
                         o3c.Tensor.from_numpy(dataset_points), knn).numpy()

    np.testing.assert_equal(indices,
                            np.array([[1, 4, 9], [1, 4, 9]], dtype=np.int32))
    assert indices.dtype == np.int32

    # Interchange query and support points.
    indices_inv = knn_search(o3c.Tensor.from_numpy(dataset_points),
                             o3c.Tensor.from_numpy(query_points), 1).numpy()

    np.testing.assert_equal(indices_inv,
                            np.array([[0] for i in range(10)], dtype=np.int32))

    # Passing same support and query points.
    indices = knn_search(o3c.Tensor.from_numpy(dataset_points),
                         o3c.Tensor.from_numpy(dataset_points), knn).numpy()
    indices_ref = np.array(
        [[0, 1, 3], [1, 0, 2], [2, 1, 5], [3, 0, 4], [4, 1, 3], [5, 2, 4],
         [6, 3, 7], [7, 4, 6], [8, 5, 7], [9, 0, 1]],
        dtype=np.int32)

    np.testing.assert_equal(indices, indices_ref)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        indices = knn_search(
            o3c.Tensor.from_numpy(np.array(query_points, dtype=np.int32)),
            o3c.Tensor.from_numpy(dataset_points), knn).numpy()

    with pytest.raises(RuntimeError):
        indices = knn_search(
            o3c.Tensor.from_numpy(query_points),
            o3c.Tensor.from_numpy(np.array(dataset_points, dtype=np.float64)),
            knn).numpy()

    # Test wrong shape.
    with pytest.raises(RuntimeError):
        indices = knn_search(o3c.Tensor.from_numpy(query_points[:, :2]),
                             o3c.Tensor.from_numpy(dataset_points),
                             knn).numpy()

    with pytest.raises(RuntimeError):
        indices = knn_search(o3c.Tensor.from_numpy(query_points),
                             o3c.Tensor.from_numpy(dataset_points[:, :2]),
                             knn).numpy()

    with pytest.raises(TypeError):
        indices = knn_search(None, o3c.Tensor.from_numpy(dataset_points[:, :2]),
                             knn).numpy()


def test_radius_search():
    query_points = np.array(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843],
         [0.064705, 0.043921, 0.087843]],
        dtype=np.float32)
    dataset_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1],
         [0.0, 0.0, 0.2], [0.0, 0.1, 0.0], [0.0, 0.1, 0.1], [0.0, 0.1, 0.2],
         [0.0, 0.2, 0.0], [0.0, 0.2, 0.1], [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=np.float32)
    radius = 0.1

    indices = radius_search(
        o3c.Tensor.from_numpy(query_points),
        o3c.Tensor.from_numpy(dataset_points),
        o3c.Tensor.from_numpy(np.array([1, 2], dtype=np.int32)),
        o3c.Tensor.from_numpy(np.array([10, 10], dtype=np.int32)),
        radius).numpy()
    np.testing.assert_equal(
        indices, np.array([[1, 4], [11, 14], [11, 14]], dtype=np.int32))
    assert indices.dtype == np.int32

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)

    indices = radius_search(
        o3c.Tensor.from_numpy(points), o3c.Tensor.from_numpy(points),
        o3c.Tensor.from_numpy(np.array([2, 3, 2], dtype=np.int32)),
        o3c.Tensor.from_numpy(np.array([3, 2, 2], dtype=np.int32)),
        11.0).numpy()

    indices_ref = np.array([[0, 1, 2], [1, 0, 2], [3, 4, -1], [3, 4, -1],
                            [4, 3, -1], [5, 6, -1], [6, 5, -1]],
                           dtype=np.int32)
    np.testing.assert_equal(indices, indices_ref)

    indices = radius_search(
        o3c.Tensor.from_numpy(points), o3c.Tensor.from_numpy(points),
        o3c.Tensor.from_numpy(np.array([1, 1, 5], dtype=np.int32)),
        o3c.Tensor.from_numpy(np.array([5, 1, 1], dtype=np.int32)),
        11.0).numpy()

    indices_ref = np.ones((7, 5), dtype=np.int32) * -1
    indices_ref[0] = [0, 1, 2, 3, 4]
    indices_ref[:, 0] = [0, 5, 6, 6, 6, 6, 6]
    np.testing.assert_equal(indices, indices_ref)

    # Test wrong dtype.
    query_batches = np.array([1, 1, 5], dtype=np.int32)
    dataset_batches = np.array([5, 1, 1], dtype=np.int32)
    with pytest.raises(RuntimeError):
        indices = radius_search(
            o3c.Tensor.from_numpy(np.array(points, dtype=np.int32)),
            o3c.Tensor.from_numpy(points), o3c.Tensor.from_numpy(query_batches),
            o3c.Tensor.from_numpy(dataset_batches), 11.0).numpy()

    with pytest.raises(RuntimeError):
        indices = radius_search(
            o3c.Tensor.from_numpy(points),
            o3c.Tensor.from_numpy(np.array(points, dtype=np.int32)),
            o3c.Tensor.from_numpy(query_batches),
            o3c.Tensor.from_numpy(dataset_batches), 11.0).numpy()

    # Test wrong shape and inconsistent batch size.
    with pytest.raises(RuntimeError):
        query_batches = np.array([2, 3, 1], dtype=np.int32)
        dataset_batches = np.array([2, 3, 2], dtype=np.int32)
        indices = radius_search(o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(query_batches),
                                o3c.Tensor.from_numpy(dataset_batches),
                                11.0).numpy()

    with pytest.raises(RuntimeError):
        query_batches = np.array([2, 3, 2], dtype=np.int32)
        dataset_batches = np.array([2, 0, 2], dtype=np.int32)
        indices = radius_search(o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(query_batches),
                                o3c.Tensor.from_numpy(dataset_batches),
                                11.0).numpy()

    # Consistent batch size, but different for both pc.
    with pytest.raises(RuntimeError):
        query_batches = np.array([2, 3, 2], dtype=np.int32)
        dataset_batches = np.array([5, 2], dtype=np.int32)
        indices = radius_search(o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(points),
                                o3c.Tensor.from_numpy(query_batches),
                                o3c.Tensor.from_numpy(dataset_batches),
                                11.0).numpy()
