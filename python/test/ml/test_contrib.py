import numpy as np
import pytest
from open3d.pybind.ml.contrib import subsample, subsample_batch


def test_subsample():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0 ,1], [1, 1, 1], [5, 0, 0], [5, 1, 0]], dtype=np.float32)
    features = np.array(range(21), dtype=np.float32).reshape(-1, 3)
    labels = np.array(range(7), dtype = np.int32)

    # Passing only points.
    sub_points_ref = np.array([[5, 0.5, 0], [0.4, 0.4, 0.4]], dtype=np.float32)
    sub_points = subsample(points, sampleDl=1.1)

    np.testing.assert_equal(sub_points, sub_points_ref)

    #Passing points and features.
    sub_features_ref = np.array([[16.5, 17.5, 18.5], [6, 7, 8]], dtype=np.float32)
    sub_points_feat = subsample(points, features=features, sampleDl=1.1)

    assert len(sub_points_feat) == 2

    sub_points, sub_features = sub_points_feat

    np.testing.assert_equal(sub_features.shape, sub_features_ref.shape)
    np.testing.assert_equal(sub_features, sub_features_ref)

    #Passing points, features and labels.
    sub_labels_ref = np.array([6, 4], dtype=np.int32)
    sub_points_feat_lab = subsample(points, features=features, classes=labels, sampleDl=1.1)

    assert len(sub_points_feat_lab) == 3

    sub_points, sub_features, sub_labels = sub_points_feat_lab

    np.testing.assert_equal(sub_labels, sub_labels_ref)
    np.testing.assert_equal(sub_features.shape, sub_features_ref.shape)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.int32), sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.float64), sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(points, features=np.array(features, dtype=np.int32), sampleDl=1.1)
    
    with pytest.raises(RuntimeError):
        sub_points = subsample(points, features=features, classes=np.array(labels, dtype=np.float32), sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(points, features=np.array(features, np.int32), classes=np.array(labels, dtype=np.float32), sampleDl=1.1)

    # Test shape mismatch
    with pytest.raises(RuntimeError):
        sub_points = subsample(points[0], sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(np.ones((10, 4), dtype=np.float32), sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(points, features=features[0], sampleDl=1.1)

    with pytest.raises(RuntimeError):
        sub_points = subsample(points[0], sampleDl=1.1)

    with pytest.raises(TypeError):
        sub_points = subsample(None, sampleDl=1.1)

def test_subsample_batch():
    points = np.ones((10, 3), dtype=np.float32)
    batches = np.array([2, 1, 3, 4], dtype=np.int32)
    features = np.ones((10, 5), dtype=np.float32)
    classes = np.ones((10,), dtype=np.int32)

    # Test handling of None
    p, b = subsample_batch(points, batches)
    p, b, f, c = subsample_batch(points, batches, features, classes)
    p, b, c = subsample_batch(points, batches, classes=classes)

# test_subsample()