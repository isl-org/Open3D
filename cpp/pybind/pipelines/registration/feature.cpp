// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/Feature.h"

#include "open3d/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/pipelines/registration/registration.h"

namespace open3d {
namespace pipelines {
namespace registration {

void pybind_feature(py::module &m) {
    // open3d.registration.Feature
    py::class_<Feature, std::shared_ptr<Feature>> feature(
            m, "Feature", "Class to store featrues for registration.");
    py::detail::bind_default_constructor<Feature>(feature);
    py::detail::bind_copy_functions<Feature>(feature);
    feature.def("resize", &Feature::Resize, "dim"_a, "n"_a,
                "Resize feature data buffer to ``dim x n``.")
            .def("dimension", &Feature::Dimension,
                 "Returns feature dimensions per point.")
            .def("num", &Feature::Num, "Returns number of points.")
            .def_readwrite("data", &Feature::data_,
                           "``dim x n`` float64 numpy array: Data buffer "
                           "storing features.")
            .def("__repr__", [](const Feature &f) {
                return std::string(
                               "Feature class with "
                               "dimension "
                               "= ") +
                       std::to_string(f.Dimension()) +
                       std::string(" and num = ") + std::to_string(f.Num()) +
                       std::string("\nAccess its data via data member.");
            });
    docstring::ClassMethodDocInject(m, "Feature", "dimension");
    docstring::ClassMethodDocInject(m, "Feature", "num");
    docstring::ClassMethodDocInject(m, "Feature", "resize",
                                    {{"dim", "Feature dimension per point."},
                                     {"n", "Number of points."}});
}

void pybind_feature_methods(py::module &m) {
    m.def("compute_fpfh_feature", &ComputeFPFHFeature,
          "Function to compute FPFH feature for a point cloud", "input"_a,
          "search_param"_a);
    docstring::FunctionDocInject(
            m, "compute_fpfh_feature",
            {{"input", "The Input point cloud."},
             {"search_param", "KDTree KNN search parameter."}});

    m.def("correspondences_from_features", &CorrespondencesFromFeatures,
          "Function to find nearest neighbor correspondences from features",
          "source_features"_a, "target_features"_a, "mutual_filter"_a = false,
          "mutual_consistency_ratio"_a = 0.1f);
    docstring::FunctionDocInject(
            m, "correspondences_from_features",
            {{"source_features", "The source features stored in (dim, N)."},
             {"target_features", "The target features stored in (dim, M)."},
             {"mutual_filter",
              "filter correspondences and return the collection of (i, j) s.t. "
              "source_features[i] and target_features[j] are mutually the "
              "nearest neighbor."},
             {"mutual_consistency_ratio",
              "Threshold to decide whether the number of filtered "
              "correspondences is sufficient. Only used when mutual_filter is "
              "enabled."}});
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
