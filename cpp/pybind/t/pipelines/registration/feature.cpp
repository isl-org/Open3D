// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
