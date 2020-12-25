import numpy as np
import pytest
from open3d.ml.contrib import subsample, subsample_batch


def compare_results_with_sorting(actual_points,
                                 expect_points,
                                 actual_features=None,
                                 expect_features=None,
                                 actual_labels=None,
                                 expect_labels=None):
    actual_argsort = actual_points.argsort(axis=0)[:, 0]
    expect_argsort = expect_points.argsort(axis=0)[:, 0]
    np.testing.assert_allclose(actual_points[actual_argsort],
                               expect_points[expect_argsort])
    if actual_features is not None and expect_features is not None:
        np.testing.assert_allclose(actual_features[actual_argsort],
                                   expect_features[expect_argsort])
    if actual_labels is not None and expect_labels is not None:
        np.testing.assert_equal(actual_labels[actual_argsort],
                                expect_labels[expect_argsort])


def compare_batched_results_with_sorting(actual_points,
                                         expect_points,
                                         actual_batches,
                                         expect_batches,
                                         actual_features=None,
                                         expect_features=None,
                                         actual_labels=None,
                                         expect_labels=None):
    np.testing.assert_equal(actual_batches, expect_batches)
    assert actual_batches.ndim == 1
    end_indices = np.cumsum(actual_batches)
    start_indices = [0] + list(end_indices[:-1])
    for s, e in zip(start_indices, end_indices):
        compare_results_with_sorting(actual_points=actual_points[s:e],
                                     expect_points=expect_points[s:e],
                                     actual_features=actual_features[s:e]
                                     if actual_features is not None else None,
                                     expect_features=expect_features[s:e]
                                     if expect_features is not None else None,
                                     actual_labels=actual_labels[s:e]
                                     if actual_labels is not None else None,
                                     expect_labels=expect_labels[s:e]
                                     if expect_labels is not None else None)


def test_subsample():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    features = np.array(range(21), dtype=np.float32).reshape(-1, 3)
    labels = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int32)

    # Reference results.
    sub_points_ref = np.array([[5, 0.5, 0], [0.4, 0.4, 0.4]], dtype=np.float32)
    sub_features_ref = np.array([[16.5, 17.5, 18.5], [6, 7, 8]],
                                dtype=np.float32)
    sub_labels_ref = np.array([1, 0], dtype=np.int32)

    # Passing only points.
    sub_points = subsample(points, sampleDl=1.1)
    compare_results_with_sorting(actual_points=sub_points,
                                 expect_points=sub_points_ref)

    # Passing points and features.
    sub_points, sub_features = subsample(points,
                                         features=features,
                                         sampleDl=1.1)
    compare_results_with_sorting(actual_points=sub_points,
                                 expect_points=sub_points_ref,
                                 actual_features=sub_features,
                                 expect_features=sub_features_ref)

    # Passing points, features and labels.
    sub_points, sub_features, sub_labels = subsample(points,
                                                     features=features,
                                                     classes=labels,
                                                     sampleDl=1.1)
    compare_results_with_sorting(actual_points=sub_points,
                                 expect_points=sub_points_ref,
                                 actual_features=sub_features,
                                 expect_features=sub_features_ref,
                                 actual_labels=sub_labels,
                                 expect_labels=sub_labels_ref)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.int32), sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(np.array(points, dtype=np.float64), sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=np.array(features, dtype=np.int32),
                               sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=features,
                               classes=np.array(labels, dtype=np.float32),
                               sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample(points,
                               features=np.array(features, np.int32),
                               classes=np.array(labels, dtype=np.float32),
                               sampleDl=1.1)

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
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                       [5, 0, 0], [5, 1, 0]],
                      dtype=np.float32)
    features = np.array(range(28), dtype=np.float32).reshape(-1, 4)
    labels = np.array([0, 0, 3, 1, 1, 2, 2], dtype=np.int32)
    batches = np.array([3, 2, 2], dtype=np.int32)

    # Reference results.
    sub_points_ref = np.array(
        [[0.3333333, 0.3333333, 0], [0.5, 0.5, 1], [5, 0.5, 0]],
        dtype=np.float32)
    sub_batch_ref = np.array([1, 1, 1], dtype=np.int32)
    sub_labels_ref = np.array([0, 1, 2], dtype=np.int32)

    # Passing only points.
    sub_points, sub_batch = subsample_batch(points, batches, sampleDl=1.1)
    compare_batched_results_with_sorting(actual_points=sub_points,
                                         expect_points=sub_points_ref,
                                         actual_batches=sub_batch,
                                         expect_batches=sub_batch_ref)

    # Passing points and features.
    sub_features_ref = np.array(
        [[4, 5, 6, 7], [14, 15, 16, 17], [22, 23, 24, 25]], dtype=np.float32)
    sub_points, sub_batch, sub_features = subsample_batch(points,
                                                          batches,
                                                          features=features,
                                                          sampleDl=1.1)
    compare_batched_results_with_sorting(actual_points=sub_points,
                                         expect_points=sub_points_ref,
                                         actual_batches=sub_batch,
                                         expect_batches=sub_batch_ref,
                                         actual_features=sub_features,
                                         expect_features=sub_features_ref)

    # Passing points, features and labels.
    sub_points, sub_batch, sub_features, sub_labels = subsample_batch(
        points, batches, features=features, classes=labels, sampleDl=1.1)
    compare_batched_results_with_sorting(actual_points=sub_points,
                                         expect_points=sub_points_ref,
                                         actual_batches=sub_batch,
                                         expect_batches=sub_batch_ref,
                                         actual_features=sub_features,
                                         expect_features=sub_features_ref,
                                         actual_labels=sub_labels,
                                         expect_labels=sub_labels_ref)

    # Test wrong dtype.
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.array(points, dtype=np.int32),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array(batches, dtype=np.float32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.array(points, dtype=np.float64),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=np.array(features,
                                                       dtype=np.int32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=features,
                                     classes=np.array(labels, dtype=np.float32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=np.array(features, np.int32),
                                     classes=np.array(labels, dtype=np.float32),
                                     sampleDl=1.1)

    # Test shape mismatch
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points[0], batches, sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(np.ones((10, 4), dtype=np.float32),
                                     batches,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     batches,
                                     features=features[0],
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points[0], batches, sampleDl=1.1)
    with pytest.raises(TypeError):
        sub_points = subsample_batch(None, None, sampleDl=1.1)

    # Test sum(batch) != num_points
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([3, 3, 2], dtype=np.int32),
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([3, 3, 2], dtype=np.int32),
                                     features=features,
                                     classes=labels,
                                     sampleDl=1.1)
    with pytest.raises(RuntimeError):
        sub_points = subsample_batch(points,
                                     np.array([1], dtype=np.int32),
                                     sampleDl=1.1)
