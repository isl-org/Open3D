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

#include "Open3D/Geometry/BoundingVolume.h"
#include "Python/docstring.h"
#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_boundingvolume(py::module &m) {
    py::class_<geometry::OrientedBoundingBox,
               PyGeometry3D<geometry::OrientedBoundingBox>,
               std::shared_ptr<geometry::OrientedBoundingBox>,
               geometry::Geometry3D>
            oriented_bounding_box(m, "OrientedBoundingBox",
                                  "Class that defines an oriented box that can "
                                  "be computed from 3D geometries.");
    py::detail::bind_default_constructor<geometry::OrientedBoundingBox>(
            oriented_bounding_box);
    py::detail::bind_copy_functions<geometry::OrientedBoundingBox>(
            oriented_bounding_box);
    oriented_bounding_box
            .def("__repr__",
                 [](const geometry::OrientedBoundingBox &box) {
                     return std::string("geometry::OrientedBoundingBox");
                 })
            .def("volume", &geometry::OrientedBoundingBox::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_box_points", &geometry::OrientedBoundingBox::GetBoxPoints,
                 "Returns the eight points that define the bounding box.")
            .def_readwrite("center", &geometry::OrientedBoundingBox::center_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("x_axis", &geometry::OrientedBoundingBox::x_axis_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("y_axis", &geometry::OrientedBoundingBox::y_axis_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("z_axis", &geometry::OrientedBoundingBox::z_axis_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("color", &geometry::OrientedBoundingBox::color_,
                           "``float64`` array of shape ``(3, )``");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "volume");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_box_points");

    py::class_<geometry::AxisAlignedBoundingBox,
               PyGeometry3D<geometry::AxisAlignedBoundingBox>,
               std::shared_ptr<geometry::AxisAlignedBoundingBox>,
               geometry::Geometry3D>
            axis_aligned_bounding_box(m, "AxisAlignedBoundingBox",
                                      "Class that defines an axis_aligned box "
                                      "that can be computed from 3D "
                                      "geometries.");
    py::detail::bind_default_constructor<geometry::AxisAlignedBoundingBox>(
            axis_aligned_bounding_box);
    py::detail::bind_copy_functions<geometry::AxisAlignedBoundingBox>(
            axis_aligned_bounding_box);
    axis_aligned_bounding_box
            .def("__repr__",
                 [](const geometry::AxisAlignedBoundingBox &box) {
                     return std::string("geometry::AxisAlignedBoundingBox");
                 })
            .def("volume", &geometry::AxisAlignedBoundingBox::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_box_points",
                 &geometry::AxisAlignedBoundingBox::GetBoxPoints,
                 "Returns the eight points that define the bounding box.")
            .def_readwrite("min_bound",
                           &geometry::AxisAlignedBoundingBox::min_bound_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("max_bound",
                           &geometry::AxisAlignedBoundingBox::max_bound_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("color", &geometry::AxisAlignedBoundingBox::color_,
                           "``float64`` array of shape ``(3, )``");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox", "volume");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_box_points");
}
