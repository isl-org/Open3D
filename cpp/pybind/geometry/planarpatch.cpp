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

#include "open3d/geometry/PlanarPatch.h"

#include <sstream>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_planarpatch(py::module &m) {
    py::class_<PlanarPatch, PyGeometry3D<PlanarPatch>,
               std::shared_ptr<PlanarPatch>, Geometry3D>
            planar_patch(m, "PlanarPatch",
                         "A planar patch in 3D, typically "
                         "detected from a point cloud.");
    py::detail::bind_default_constructor<PlanarPatch>(planar_patch);
    py::detail::bind_copy_functions<PlanarPatch>(planar_patch);
    planar_patch
            .def("__repr__",
                 [](const PlanarPatch &patch) {
                     std::stringstream s;
                     const Eigen::Vector3d n = patch.normal_;
                     const double d = patch.dist_from_origin_;
                     s << "PlanarPatch: (n, d) = (" << n.x() << ", " << n.y()
                       << ", " << n.z() << ", " << d << ")";
                     return s.str();
                 })
            .def_readwrite("center", &PlanarPatch::center_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("normal", &PlanarPatch::normal_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("dist_from_origin", &PlanarPatch::dist_from_origin_,
                           "``float64`` scalar")
            .def_readwrite("basis_x", &PlanarPatch::basis_x_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("basis_y", &PlanarPatch::basis_y_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("color", &PlanarPatch::color_,
                           "``float64`` array of shape ``(3, )``");
}

}  // namespace geometry
}  // namespace open3d
