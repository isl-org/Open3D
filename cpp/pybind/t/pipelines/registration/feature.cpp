// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Feature.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"
#include "pybind/t/pipelines/registration/registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

void pybind_feature(py::module &m) {
    m.def("compute_fpfh_feature", &ComputeFPFHFeature,
          py::call_guard<py::gil_scoped_release>(),
          R"(Function to compute FPFH feature for a point cloud.
It uses KNN search (Not recommended to use on GPU) if only max_nn parameter
is provided, Radius search (Not recommended to use on GPU) if only radius
parameter is provided, and Hybrid search (Recommended) if both are provided.)",
          "input"_a, "max_nn"_a = 100, "radius"_a = py::none());
    docstring::FunctionDocInject(
            m, "compute_fpfh_feature",
            {{"input",
              "The input point cloud with data type float32 or float64."},
             {"max_nn",
              "[optional] Neighbor search max neighbors parameter.[Default = "
              "100]"},
             {"radius",
              "[optional] Neighbor search radius parameter. [Recommended ~5x "
              "voxel size]"}});

    m.def("correspondences_from_features", &CorrespondencesFromFeatures,
          py::call_guard<py::gil_scoped_release>(),
          R"(Function to query nearest neighbors of source_features in target_features.)",
          "source_features"_a, "target_features"_a, "mutual_filter"_a = false,
          "mutual_consistency_ratio"_a = 0.1f);
    docstring::FunctionDocInject(
            m, "correspondences_from_features",
            {{"source_features", "The source features in shape (N, dim)."},
             {"target_features", "The target features in shape (M, dim)."},
             {"mutual_filter",
              "filter correspondences and return the collection of (i, j) "
              "s.t. "
              "source_features[i] and target_features[j] are mutually the "
              "nearest neighbor."},
             {"mutual_consistency_ratio",
              "Threshold to decide whether the number of filtered "
              "correspondences is sufficient. Only used when "
              "mutual_filter is "
              "enabled."}});
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
