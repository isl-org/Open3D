import numpy as np
from open3d.pybind.ml.contrib import subsample, subsample_batch


def test_subsample():
    points = np.ones((10, 3), dtype=np.float32)
    features = np.ones((10, 5), dtype=np.float32)
    classes = np.ones((10,), dtype=np.int32)

    # Test handling of None
    p = subsample(points)
    p, f, c = subsample(points, features, classes)
    p, c = subsample(points, classes=classes)


def test_subsample_batch():
    points = np.ones((10, 3), dtype=np.float32)
    batches = np.array([2, 1, 3, 4], dtype=np.int32)
    features = np.ones((10, 5), dtype=np.float32)
    classes = np.ones((10,), dtype=np.int32)

    # Test handling of None
    p, b = subsample_batch(points, batches)
    p, b, f, c = subsample_batch(points, batches, features, classes)
    p, b, c = subsample_batch(points, batches, classes=classes)
