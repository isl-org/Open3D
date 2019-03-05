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

#include "Python/registration/registration.h"

#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Registration/Feature.h>

using namespace open3d;

void pybind_feature(py::module &m) {
    py::class_<registration::Feature, std::shared_ptr<registration::Feature>>
            feature(m, "Feature", "Feature");
    py::detail::bind_default_constructor<registration::Feature>(feature);
    py::detail::bind_copy_functions<registration::Feature>(feature);
    feature.def("resize", &registration::Feature::Resize, "dim"_a, "n"_a)
            .def("dimension", &registration::Feature::Dimension)
            .def("num", &registration::Feature::Num)
            .def_readwrite("data", &registration::Feature::data_)
            .def("__repr__", [](const registration::Feature &f) {
                return std::string(
                               "registration::Feature class with dimension "
                               "= ") +
                       std::to_string(f.Dimension()) +
                       std::string(" and num = ") + std::to_string(f.Num()) +
                       std::string("\nAccess its data via data member.");
            });
}

void pybind_feature_methods(py::module &m) {
    m.def("compute_fpfh_feature", &registration::ComputeFPFHFeature,
          "Function to compute FPFH feature for a point cloud", "input"_a,
          "search_param"_a);
}
