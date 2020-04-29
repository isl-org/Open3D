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

#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/PointCloud.h"

#include "open3d_pybind/docstring.h"
#include "open3d_pybind/geometry/geometry.h"
#include "open3d_pybind/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_lineset(py::module &m) {
    py::class_<geometry::LineSet, PyGeometry3D<geometry::LineSet>,
               std::shared_ptr<geometry::LineSet>, geometry::Geometry3D>
            lineset(m, "LineSet",
                    "LineSet define a sets of lines in 3D. A typical "
                    "application is to display the point cloud correspondence "
                    "pairs.");
    py::detail::bind_default_constructor<geometry::LineSet>(lineset);
    py::detail::bind_copy_functions<geometry::LineSet>(lineset);
    lineset.def(py::init<const std::vector<Eigen::Vector3d> &,
                         const std::vector<Eigen::Vector2i> &>(),
                "Create a LineSet from given points and line indices",
                "points"_a, "lines"_a)
            .def("__repr__",
                 [](const geometry::LineSet &lineset) {
                     return std::string("geometry::LineSet with ") +
                            std::to_string(lineset.lines_.size()) + " lines.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_points", &geometry::LineSet::HasPoints,
                 "Returns ``True`` if the object contains points.")
            .def("has_lines", &geometry::LineSet::HasLines,
                 "Returns ``True`` if the object contains lines.")
            .def("has_colors", &geometry::LineSet::HasColors,
                 "Returns ``True`` if the object's lines contain "
                 "colors.")
            .def("get_line_coordinate", &geometry::LineSet::GetLineCoordinate,
                 "line_index"_a)
            .def("paint_uniform_color", &geometry::LineSet::PaintUniformColor,
                 "Assigns each line in the line set the same color.", "color"_a)
            .def_static("create_from_point_cloud_correspondences",
                        &geometry::LineSet::CreateFromPointCloudCorrespondences,
                        "Factory function to create a LineSet from two "
                        "pointclouds and a correspondence set.",
                        "cloud0"_a, "cloud1"_a, "correspondences"_a)
            .def_static("create_from_oriented_bounding_box",
                        &geometry::LineSet::CreateFromOrientedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "OrientedBoundingBox.",
                        "box"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &geometry::LineSet::CreateFromAxisAlignedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "AxisAlignedBoundingBox.",
                        "box"_a)
            .def_static("create_from_triangle_mesh",
                        &geometry::LineSet::CreateFromTriangleMesh,
                        "Factory function to create a LineSet from edges of a "
                        "triangle mesh.",
                        "mesh"_a)
            .def_static("create_from_tetra_mesh",
                        &geometry::LineSet::CreateFromTetraMesh,
                        "Factory function to create a LineSet from edges of a "
                        "tetra mesh.",
                        "mesh"_a)
            .def_readwrite("points", &geometry::LineSet::points_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "coordinates.")
            .def_readwrite("lines", &geometry::LineSet::lines_,
                           "``int`` array of shape ``(num_lines, 2)``, use "
                           "``numpy.asarray()`` to access data: Lines denoted "
                           "by the index of points forming the line.")
            .def_readwrite(
                    "colors", &geometry::LineSet::colors_,
                    "``float64`` array of shape ``(num_lines, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of lines.");
    docstring::ClassMethodDocInject(m, "LineSet", "has_colors");
    docstring::ClassMethodDocInject(m, "LineSet", "has_lines");
    docstring::ClassMethodDocInject(m, "LineSet", "has_points");
    docstring::ClassMethodDocInject(m, "LineSet", "get_line_coordinate",
                                    {{"line_index", "Index of the line."}});
    docstring::ClassMethodDocInject(m, "LineSet", "paint_uniform_color",
                                    {{"color", "Color for the LineSet."}});
    docstring::ClassMethodDocInject(
            m, "LineSet", "create_from_point_cloud_correspondences",
            {{"cloud0", "First point cloud."},
             {"cloud1", "Second point cloud."},
             {"correspondences", "Set of correspondences."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_oriented_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_axis_aligned_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet", "create_from_triangle_mesh",
                                    {{"mesh", "The input triangle mesh."}});
    docstring::ClassMethodDocInject(m, "LineSet", "create_from_tetra_mesh",
                                    {{"mesh", "The input tetra mesh."}});
}

void pybind_lineset_methods(py::module &m) {}
