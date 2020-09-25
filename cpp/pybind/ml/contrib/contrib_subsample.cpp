// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/GridSubsampling.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/ml/contrib/contrib.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

const py::tuple SubsampleBatch(py::array points,
                               py::array batches,
                               utility::optional<py::array> features,
                               utility::optional<py::array> classes,
                               float sampleDl,
                               const std::string& method,
                               int max_p,
                               int verbose) {
    std::vector<PointXYZ> original_points;
    std::vector<PointXYZ> subsampled_points;
    std::vector<int> original_batches;
    std::vector<int> subsampled_batches;
    std::vector<float> original_features;
    std::vector<float> subsampled_features;
    std::vector<int> original_classes;
    std::vector<int> subsampled_classes;

    // Fill original_points.
    core::Tensor points_t = core::PyArrayToTensor(points, true).Contiguous();
    if (points_t.GetDtype() != core::Dtype::Float32) {
        utility::LogError("points must be np.float32.");
    }
    if (points_t.NumDims() != 2 || points_t.GetShape()[1] != 3) {
        utility::LogError("points must have shape (N, 3), but got {}.",
                          points_t.GetShape().ToString());
    }
    int64_t num_points = points_t.NumElements() / 3;
    original_points = std::vector<PointXYZ>(
            reinterpret_cast<PointXYZ*>(points_t.GetDataPtr()),
            reinterpret_cast<PointXYZ*>(points_t.GetDataPtr()) + num_points);

    // Fill original batches.
    core::Tensor batches_t = core::PyArrayToTensor(batches, true).Contiguous();
    if (batches_t.GetDtype() != core::Dtype::Int32) {
        utility::LogError("batches must be np.int32.");
    }
    if (batches_t.NumDims() != 1) {
        utility::LogError("batches must have shape (NB,), but got {}.",
                          batches_t.GetShape().ToString());
    }
    int64_t num_batches = batches_t.GetShape()[0];
    if (static_cast<int64_t>(batches_t.Sum({0}).Item<int32_t>()) !=
        num_points) {
        utility::LogError("batches got {} points, but points got {} points.",
                          batches_t.Sum({0}).Item<int32_t>(), num_points);
    }
    original_batches = batches_t.ToFlatVector<int32_t>();
    if (verbose) {
        utility::LogInfo("Got {} batches with a total of {} points as inputs.",
                         num_batches, num_points);
    }

    // Fill original_features.
    int64_t feature_dim = -1;
    if (features.has_value()) {
        core::Tensor features_t =
                core::PyArrayToTensor(features.value(), true).Contiguous();
        if (features_t.GetDtype() != core::Dtype::Float32) {
            utility::LogError("features must be np.float32.");
        }
        if (features_t.NumDims() != 2) {
            utility::LogError("features must have shape (N, d), but got {}.",
                              features_t.GetShape().ToString());
        }
        if (features_t.GetShape()[0] != num_points) {
            utility::LogError(
                    "features's shape {} is not compatible with "
                    "points's shape {}, their first dimension must "
                    "be equal.",
                    features_t.GetShape().ToString(),
                    points_t.GetShape().ToString());
        }
        feature_dim = features_t.GetShape()[1];
        original_features = features_t.ToFlatVector<float>();
    }

    // Fill original_classes.
    if (classes.has_value()) {
        core::Tensor classes_t =
                core::PyArrayToTensor(classes.value(), true).Contiguous();
        if (classes_t.GetDtype() != core::Dtype::Int32) {
            utility::LogError("classes must be np.int32.");
        }
        if (classes_t.NumDims() != 1) {
            utility::LogError("classes must have shape (N,), but got {}.",
                              classes_t.GetShape().ToString());
        }
        if (classes_t.GetShape()[0] != num_points) {
            utility::LogError(
                    "classes's shape {} is not compatible with "
                    "points's shape {}, their first dimension must "
                    "be equal.",
                    classes_t.GetShape().ToString(),
                    points_t.GetShape().ToString());
        }
        original_classes = classes_t.ToFlatVector<int32_t>();
    }

    // Call function.
    batch_grid_subsampling(
            original_points, subsampled_points, original_features,
            subsampled_features, original_classes, subsampled_classes,
            original_batches, subsampled_batches, sampleDl, max_p);

    // Wrap result subsampled_points. Data will be copied.
    assert(std::is_pod<PointXYZ>());
    int64_t num_subsampled_points =
            static_cast<int64_t>(subsampled_points.size());
    core::Tensor subsampled_points_t(
            reinterpret_cast<float*>(subsampled_points.data()),
            {num_subsampled_points, 3}, core::Dtype::Float32);

    // Wrap result subsampled_batches. Data will be copied.
    int64_t num_subsampled_batches =
            static_cast<int64_t>(subsampled_batches.size());
    core::Tensor subsampled_batches_t = core::Tensor(
            subsampled_batches, {num_subsampled_batches}, core::Dtype::Int32);
    if (static_cast<int64_t>(subsampled_batches_t.Sum({0}).Item<int32_t>()) !=
        num_subsampled_points) {
        utility::LogError(
                "subsampled_batches got {} points, but subsampled_points got "
                "{} points.",
                subsampled_batches_t.Sum({0}).Item<int32_t>(),
                num_subsampled_points);
    }
    if (verbose) {
        utility::LogInfo("Subsampled to {} batches with a total of {} points.",
                         num_subsampled_batches, num_subsampled_points);
    }

    // Wrap result subsampled_features. Data will be copied.
    core::Tensor subsampled_features_t;
    if (features.has_value()) {
        if (subsampled_features.size() % num_subsampled_points != 0) {
            utility::LogError(
                    "Error: subsampled_points.size() {} is not a "
                    "multiple of num_subsampled_points {}.",
                    subsampled_points.size(), num_subsampled_points);
        }
        int64_t subsampled_feature_dim =
                static_cast<int64_t>(subsampled_features.size()) /
                num_subsampled_points;
        if (feature_dim != subsampled_feature_dim) {
            utility::LogError(
                    "Error: input feature dim {} does not match "
                    "the subsampled feature dim {}.",
                    feature_dim, subsampled_feature_dim);
        }
        subsampled_features_t = core::Tensor(
                subsampled_features, {num_subsampled_points, feature_dim},
                core::Dtype::Float32);
    }

    // Wrap result subsampled_classes. Data will be copied.
    core::Tensor subsampled_classes_t;
    if (classes.has_value()) {
        if (subsampled_classes.size() != num_subsampled_points) {
            utility::LogError(
                    "Error: subsampled_classes.size() {} != "
                    "num_subsampled_points {}.",
                    subsampled_classes.size(), num_subsampled_points);
        }
        subsampled_classes_t =
                core::Tensor(subsampled_classes, {num_subsampled_points},
                             core::Dtype::Int32);
    }

    if (features.has_value() && classes.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_batches_t),
                              core::TensorToPyArray(subsampled_features_t),
                              core::TensorToPyArray(subsampled_classes_t));
    } else if (features.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_batches_t),
                              core::TensorToPyArray(subsampled_features_t));
    } else if (classes.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_batches_t),
                              core::TensorToPyArray(subsampled_classes_t));
    } else {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_batches_t));
    }
}

const py::object Subsample(py::array points,
                           utility::optional<py::array> features,
                           utility::optional<py::array> classes,
                           float sampleDl,
                           int verbose) {
    std::vector<PointXYZ> original_points;
    std::vector<PointXYZ> subsampled_points;
    std::vector<float> original_features;
    std::vector<float> subsampled_features;
    std::vector<int> original_classes;
    std::vector<int> subsampled_classes;

    // Fill original_points.
    core::Tensor points_t = core::PyArrayToTensor(points, true).Contiguous();
    if (points_t.GetDtype() != core::Dtype::Float32) {
        utility::LogError("points must be np.float32.");
    }
    if (points_t.NumDims() != 2 || points_t.GetShape()[1] != 3) {
        utility::LogError("points must have shape (N, 3), but got {}.",
                          points_t.GetShape().ToString());
    }
    int64_t num_points = points_t.NumElements() / 3;
    original_points = std::vector<PointXYZ>(
            reinterpret_cast<PointXYZ*>(points_t.GetDataPtr()),
            reinterpret_cast<PointXYZ*>(points_t.GetDataPtr()) + num_points);
    if (verbose) {
        utility::LogInfo("Got {} points as inputs.", num_points);
    }

    // Fill original_features.
    int64_t feature_dim = -1;
    if (features.has_value()) {
        core::Tensor features_t =
                core::PyArrayToTensor(features.value(), true).Contiguous();
        if (features_t.GetDtype() != core::Dtype::Float32) {
            utility::LogError("features must be np.float32.");
        }
        if (features_t.NumDims() != 2) {
            utility::LogError("features must have shape (N, d), but got {}.",
                              features_t.GetShape().ToString());
        }
        if (features_t.GetShape()[0] != num_points) {
            utility::LogError(
                    "features's shape {} is not compatible with "
                    "points's shape {}, their first dimension must "
                    "be equal.",
                    features_t.GetShape().ToString(),
                    points_t.GetShape().ToString());
        }
        feature_dim = features_t.GetShape()[1];
        original_features = features_t.ToFlatVector<float>();
    }

    // Fill original_classes.
    if (classes.has_value()) {
        core::Tensor classes_t =
                core::PyArrayToTensor(classes.value(), true).Contiguous();
        if (classes_t.GetDtype() != core::Dtype::Int32) {
            utility::LogError("classes must be np.int32.");
        }
        if (classes_t.NumDims() != 1) {
            utility::LogError("classes must have shape (N,), but got {}.",
                              classes_t.GetShape().ToString());
        }
        if (classes_t.GetShape()[0] != num_points) {
            utility::LogError(
                    "classes's shape {} is not compatible with "
                    "points's shape {}, their first dimension must "
                    "be equal.",
                    classes_t.GetShape().ToString(),
                    points_t.GetShape().ToString());
        }
        original_classes = classes_t.ToFlatVector<int32_t>();
    }

    // Call function.
    grid_subsampling(original_points, subsampled_points, original_features,
                     subsampled_features, original_classes, subsampled_classes,
                     sampleDl, verbose);

    // Wrap result subsampled_points. Data will be copied.
    assert(std::is_pod<PointXYZ>());
    int64_t num_subsampled_points =
            static_cast<int64_t>(subsampled_points.size());
    core::Tensor subsampled_points_t(
            reinterpret_cast<float*>(subsampled_points.data()),
            {num_subsampled_points, 3}, core::Dtype::Float32);
    if (verbose) {
        utility::LogInfo("Subsampled to {} points.", num_subsampled_points);
    }

    // Wrap result subsampled_features. Data will be copied.
    core::Tensor subsampled_features_t;
    if (features.has_value()) {
        if (subsampled_features.size() % num_subsampled_points != 0) {
            utility::LogError(
                    "Error: subsampled_points.size() {} is not a "
                    "multiple of num_subsampled_points {}.",
                    subsampled_points.size(), num_subsampled_points);
        }
        int64_t subsampled_feature_dim =
                static_cast<int64_t>(subsampled_features.size()) /
                num_subsampled_points;
        if (feature_dim != subsampled_feature_dim) {
            utility::LogError(
                    "Error: input feature dim {} does not match "
                    "the subsampled feature dim {}.",
                    feature_dim, subsampled_feature_dim);
        }
        subsampled_features_t = core::Tensor(
                subsampled_features, {num_subsampled_points, feature_dim},
                core::Dtype::Float32);
    }

    // Wrap result subsampled_classes. Data will be copied.
    core::Tensor subsampled_classes_t;
    if (classes.has_value()) {
        if (subsampled_classes.size() != num_subsampled_points) {
            utility::LogError(
                    "Error: subsampled_classes.size() {} != "
                    "num_subsampled_points {}.",
                    subsampled_classes.size(), num_subsampled_points);
        }
        subsampled_classes_t =
                core::Tensor(subsampled_classes, {num_subsampled_points},
                             core::Dtype::Int32);
    }

    if (features.has_value() && classes.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_features_t),
                              core::TensorToPyArray(subsampled_classes_t));
    } else if (features.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_features_t));
    } else if (classes.has_value()) {
        return py::make_tuple(core::TensorToPyArray(subsampled_points_t),
                              core::TensorToPyArray(subsampled_classes_t));
    } else {
        return core::TensorToPyArray(subsampled_points_t);
    }
}

void pybind_contrib_subsample(py::module& m_contrib) {
    m_contrib.def("subsample", &Subsample, "points"_a,
                  "features"_a = py::none(), "classes"_a = py::none(),
                  "sampleDl"_a = 0.1, "verbose"_a = 0);

    m_contrib.def("subsample_batch", &SubsampleBatch, "points"_a, "batches"_a,
                  "features"_a = py::none(), "classes"_a = py::none(),
                  "sampleDl"_a = 0.1, "method"_a = "barycenters", "max_p"_a = 0,
                  "verbose"_a = 0);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
