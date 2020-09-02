import open3d.core as o3c
import numpy as np
import pytest
from open3d.pybind.ml.contrib import knn_search, radius_search


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
    query_batches = np.array([1, 2], dtype=np.int32)
    dataset_batches = np.array([10, 10], dtype=np.int32)
    radius = 0.1

    indices = radius_search(o3c.Tensor.from_numpy(query_points),
                            o3c.Tensor.from_numpy(dataset_points),
                            o3c.Tensor.from_numpy(query_batches),
                            o3c.Tensor.from_numpy(dataset_batches), radius)
    print(indices)
