// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/BoundingVolume.h"

#include <sstream>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_boundingvolume_declarations(py::module &m) {
    py::class_<OrientedBoundingBox, PyGeometry3D<OrientedBoundingBox>,
               std::shared_ptr<OrientedBoundingBox>, Geometry3D>
            oriented_bounding_box(m, "OrientedBoundingBox",
                                  "Class that defines an oriented box that can "
                                  "be computed from 3D geometries.");
    py::class_<AxisAlignedBoundingBox, PyGeometry3D<AxisAlignedBoundingBox>,
               std::shared_ptr<AxisAlignedBoundingBox>, Geometry3D>
            axis_aligned_bounding_box(m, "AxisAlignedBoundingBox",
                                      "Class that defines an axis_aligned box "
                                      "that can be computed from 3D "
                                      "geometries, The axis aligned bounding "
                                      "box uses the coordinate axes for "
                                      "bounding box generation.");
}
void pybind_boundingvolume_definitions(py::module &m) {
    auto oriented_bounding_box = static_cast<
            py::class_<OrientedBoundingBox, PyGeometry3D<OrientedBoundingBox>,
                       std::shared_ptr<OrientedBoundingBox>, Geometry3D>>(
            m.attr("OrientedBoundingBox"));
    py::detail::bind_default_constructor<OrientedBoundingBox>(
            oriented_bounding_box);
    py::detail::bind_copy_functions<OrientedBoundingBox>(oriented_bounding_box);
    oriented_bounding_box
            .def(py::init<const Eigen::Vector3d &, const Eigen::Matrix3d &,
                          const Eigen::Vector3d &>(),
                 "Create OrientedBoudingBox from center, rotation R and extent "
                 "in x, y and z "
                 "direction",
                 "center"_a, "R"_a, "extent"_a)
            .def("__repr__",
                 [](const OrientedBoundingBox &box) {
                     std::stringstream s;
                     auto c = box.center_;
                     auto e = box.extent_;
                     s << "OrientedBoundingBox: center: (" << c.x() << ", "
                       << c.y() << ", " << c.z() << "), extent: " << e.x()
                       << ", " << e.y() << ", " << e.z() << ")";
                     return s.str();
                 })
            .def("get_point_indices_within_bounding_box",
                 &OrientedBoundingBox::GetPointIndicesWithinBoundingBox,
                 "Return indices to points that are within the bounding box.",
                 "points"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &OrientedBoundingBox::CreateFromAxisAlignedBoundingBox,
                        "Returns an oriented bounding box from the "
                        "AxisAlignedBoundingBox.",
                        "aabox"_a)
            .def_static("create_from_points",
                        &OrientedBoundingBox::CreateFromPoints, "points"_a,
                        "robust"_a = false,
                        R"doc(
Creates the oriented bounding box that encloses the set of points.

Computes the oriented bounding box based on the PCA of the convex hull.
The returned bounding box is an approximation to the minimal bounding box.

Args:
     points (open3d.utility.Vector3dVector): Input points.
     robust (bool): If set to true uses a more robust method which works in
          degenerate cases but introduces noise to the points coordinates.

Returns:
     open3d.geometry.OrientedBoundingBox: The oriented bounding box. The
     bounding box is oriented such that the axes are ordered with respect to
     the principal components.
)doc")
            .def_static("create_from_points_minimal",
                        &OrientedBoundingBox::CreateFromPointsMinimal,
                        "points"_a, "robust"_a = false,
                        R"doc(
Creates the oriented bounding box with the smallest volume.

The algorithm makes use of the fact that at least one edge of
the convex hull must be collinear with an edge of the minimum
bounding box: for each triangle in the convex hull, calculate
the minimal axis aligned box in the frame of that triangle.
at the end, return the box with the smallest volume

Args:
     points (open3d.utility.Vector3dVector): Input points.
     robust (bool): If set to true uses a more robust method which works in
          degenerate cases but introduces noise to the points coordinates.

Returns:
     open3d.geometry.OrientedBoundingBox: The oriented bounding box. The
     bounding box is oriented such that its volume is minimized.
)doc")
            .def("volume", &OrientedBoundingBox::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_box_points", &OrientedBoundingBox::GetBoxPoints,
                 "Returns the eight points that define the bounding box.")
            .def_readwrite("center", &OrientedBoundingBox::center_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("R", &OrientedBoundingBox::R_,
                           "``float64`` array of shape ``(3,3 )``")
            .def_readwrite("extent", &OrientedBoundingBox::extent_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("color", &OrientedBoundingBox::color_,
                           "``float64`` array of shape ``(3, )``");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "volume");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_box_points");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox",
                                    "get_point_indices_within_bounding_box",
                                    {{"points", "A list of points."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "create_from_axis_aligned_bounding_box",
            {{"aabox",
              "AxisAlignedBoundingBox object from which OrientedBoundingBox is "
              "created."}});

    auto axis_aligned_bounding_box = static_cast<py::class_<
            AxisAlignedBoundingBox, PyGeometry3D<AxisAlignedBoundingBox>,
            std::shared_ptr<AxisAlignedBoundingBox>, Geometry3D>>(
            m.attr("AxisAlignedBoundingBox"));
    py::detail::bind_default_constructor<AxisAlignedBoundingBox>(
            axis_aligned_bounding_box);
    py::detail::bind_copy_functions<AxisAlignedBoundingBox>(
            axis_aligned_bounding_box);
    axis_aligned_bounding_box
            .def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(),
                 "Create an AxisAlignedBoundingBox from min bounds and max "
                 "bounds in x, y and z",
                 "min_bound"_a, "max_bound"_a)
            .def("__repr__",
                 [](const AxisAlignedBoundingBox &box) {
                     std::stringstream s;
                     auto mn = box.min_bound_;
                     auto mx = box.max_bound_;
                     s << "AxisAlignedBoundingBox: min: (" << mn.x() << ", "
                       << mn.y() << ", " << mn.z() << "), max: (" << mx.x()
                       << ", " << mx.y() << ", " << mx.z() << ")";
                     return s.str();
                 })
            .def(py::self += py::self)
            .def("volume", &AxisAlignedBoundingBox::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_box_points", &AxisAlignedBoundingBox::GetBoxPoints,
                 "Returns the eight points that define the bounding box.")
            .def("get_extent", &AxisAlignedBoundingBox::GetExtent,
                 "Get the extent/length of the bounding box in x, y, and z "
                 "dimension.")
            .def("get_half_extent", &AxisAlignedBoundingBox::GetHalfExtent,
                 "Returns the half extent of the bounding box.")
            .def("get_max_extent", &AxisAlignedBoundingBox::GetMaxExtent,
                 "Returns the maximum extent, i.e. the maximum of X, Y and Z "
                 "axis")
            .def("get_point_indices_within_bounding_box",
                 &AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox,
                 "Return indices to points that are within the bounding box.",
                 "points"_a)
            .def("get_print_info", &AxisAlignedBoundingBox::GetPrintInfo,
                 "Returns the 3D dimensions of the bounding box in string "
                 "format.")
            .def_static(
                    "create_from_points",
                    &AxisAlignedBoundingBox::CreateFromPoints,
                    "Creates the bounding box that encloses the set of points.",
                    "points"_a)
            .def_readwrite("min_bound", &AxisAlignedBoundingBox::min_bound_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("max_bound", &AxisAlignedBoundingBox::max_bound_,
                           "``float64`` array of shape ``(3, )``")
            .def_readwrite("color", &AxisAlignedBoundingBox::color_,
                           "``float64`` array of shape ``(3, )``");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox", "volume");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_box_points");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox", "get_extent");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_half_extent");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_max_extent");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_point_indices_within_bounding_box",
                                    {{"points", "A list of points."}});
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "get_print_info");
    docstring::ClassMethodDocInject(m, "AxisAlignedBoundingBox",
                                    "create_from_points",
                                    {{"points", "A list of points."}});
}

}  // namespace geometry
}  // namespace open3d
