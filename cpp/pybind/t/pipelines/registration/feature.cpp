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
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
