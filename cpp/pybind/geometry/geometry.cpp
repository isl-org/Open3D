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

#include "open3d/geometry/Geometry.h"

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_geometry_classes(py::module &m) {
    // open3d.geometry functions
    m.def("get_rotation_matrix_from_xyz", &Geometry3D::GetRotationMatrixFromXYZ,
          "rotation"_a);
    m.def("get_rotation_matrix_from_yzx", &Geometry3D::GetRotationMatrixFromYZX,
          "rotation"_a);
    m.def("get_rotation_matrix_from_zxy", &Geometry3D::GetRotationMatrixFromZXY,
          "rotation"_a);
    m.def("get_rotation_matrix_from_xzy", &Geometry3D::GetRotationMatrixFromXZY,
          "rotation"_a);
    m.def("get_rotation_matrix_from_zyx", &Geometry3D::GetRotationMatrixFromZYX,
          "rotation"_a);
    m.def("get_rotation_matrix_from_yxz", &Geometry3D::GetRotationMatrixFromYXZ,
          "rotation"_a);
    m.def("get_rotation_matrix_from_axis_angle",
          &Geometry3D::GetRotationMatrixFromAxisAngle, "rotation"_a);
    m.def("get_rotation_matrix_from_quaternion",
          &Geometry3D::GetRotationMatrixFromQuaternion, "rotation"_a);

    // open3d.geometry.Geometry
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");
    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.")
            .def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.")
            .def("get_geometry_type", &Geometry::GetGeometryType,
                 "Returns one of registered geometry types.")
            .def("dimension", &Geometry::Dimension,
                 "Returns whether the geometry is 2D or 3D.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
    docstring::ClassMethodDocInject(m, "Geometry", "get_geometry_type");
    docstring::ClassMethodDocInject(m, "Geometry", "dimension");

    // open3d.geometry.Geometry.Type
    py::enum_<Geometry::GeometryType> geometry_type(geometry, "Type",
                                                    py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    geometry_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Geometry types.";
            }),
            py::none(), py::none(), "");

    geometry_type.value("Unspecified", Geometry::GeometryType::Unspecified)
            .value("PointCloud", Geometry::GeometryType::PointCloud)
            .value("VoxelGrid", Geometry::GeometryType::VoxelGrid)
            .value("LineSet", Geometry::GeometryType::LineSet)
            .value("TriangleMesh", Geometry::GeometryType::TriangleMesh)
            .value("HalfEdgeTriangleMesh",
                   Geometry::GeometryType::HalfEdgeTriangleMesh)
            .value("Image", Geometry::GeometryType::Image)
            .value("RGBDImage", Geometry::GeometryType::RGBDImage)
            .value("TetraMesh", Geometry::GeometryType::TetraMesh)
            .export_values();

    py::class_<Geometry3D, PyGeometry3D<Geometry3D>,
               std::shared_ptr<Geometry3D>, Geometry>
            geometry3d(m, "Geometry3D",
                       "The base geometry class for 3D geometries.");
    geometry3d
            .def("get_min_bound", &Geometry3D::GetMinBound,
                 "Returns min bounds for geometry coordinates.")
            .def("get_max_bound", &Geometry3D::GetMaxBound,
                 "Returns max bounds for geometry coordinates.")
            .def("get_center", &Geometry3D::GetCenter,
                 "Returns the center of the geometry coordinates.")
            .def("get_axis_aligned_bounding_box",
                 &Geometry3D::GetAxisAlignedBoundingBox,
                 "Returns an axis-aligned bounding box of the geometry.")
            .def("get_oriented_bounding_box",
                 &Geometry3D::GetOrientedBoundingBox, "robust"_a = false,
                 R"doc(
Returns the oriented bounding box for the geometry.

Computes the oriented bounding box based on the PCA of the convex hull.
The returned bounding box is an approximation to the minimal bounding box.

Args:
     robust (bool): If set to true uses a more robust method which works in 
          degenerate cases but introduces noise to the points coordinates.

Returns:
     open3d.geometry.OrientedBoundingBox: The oriented bounding box. The
     bounding box is oriented such that the axes are ordered with respect to
     the principal components.
)doc")
            .def("transform", &Geometry3D::Transform,
                 "Apply transformation (4x4 matrix) to the geometry "
                 "coordinates.")
            .def("translate", &Geometry3D::Translate,
                 "Apply translation to the geometry coordinates.",
                 "translation"_a, "relative"_a = true)
            .def("scale",
                 (Geometry3D &
                  (Geometry3D::*)(const double, const Eigen::Vector3d &)) &
                         Geometry3D::Scale,
                 "Apply scaling to the geometry coordinates.", "scale"_a,
                 "center"_a)
            .def("scale", &Geometry3D::Scale,
                 "Apply scaling to the geometry coordinates.", "scale"_a,
                 "center"_a)
            .def("rotate",
                 py::overload_cast<const Eigen::Matrix3d &>(
                         &Geometry3D::Rotate),
                 "Apply rotation to the geometry coordinates and normals.",
                 "R"_a)
            .def("rotate",
                 py::overload_cast<const Eigen::Matrix3d &,
                                   const Eigen::Vector3d &>(
                         &Geometry3D::Rotate),
                 "Apply rotation to the geometry coordinates and normals.",
                 "R"_a, "center"_a)
            .def_static("get_rotation_matrix_from_xyz",
                        &Geometry3D::GetRotationMatrixFromXYZ, "rotation"_a)
            .def_static("get_rotation_matrix_from_yzx",
                        &Geometry3D::GetRotationMatrixFromYZX, "rotation"_a)
            .def_static("get_rotation_matrix_from_zxy",
                        &Geometry3D::GetRotationMatrixFromZXY, "rotation"_a)
            .def_static("get_rotation_matrix_from_xzy",
                        &Geometry3D::GetRotationMatrixFromXZY, "rotation"_a)
            .def_static("get_rotation_matrix_from_zyx",
                        &Geometry3D::GetRotationMatrixFromZYX, "rotation"_a)
            .def_static("get_rotation_matrix_from_yxz",
                        &Geometry3D::GetRotationMatrixFromYXZ, "rotation"_a)
            .def_static("get_rotation_matrix_from_axis_angle",
                        &Geometry3D::GetRotationMatrixFromAxisAngle,
                        "rotation"_a)
            .def_static("get_rotation_matrix_from_quaternion",
                        &Geometry3D::GetRotationMatrixFromQuaternion,
                        "rotation"_a);
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_min_bound");
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_max_bound");
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_center");
    docstring::ClassMethodDocInject(m, "Geometry3D",
                                    "get_axis_aligned_bounding_box");
    docstring::ClassMethodDocInject(m, "Geometry3D", "transform");
    docstring::ClassMethodDocInject(
            m, "Geometry3D", "translate",
            {{"translation", "A 3D vector to transform the geometry"},
             {"relative",
              "If true, the translation vector is directly added to the "
              "geometry "
              "coordinates. Otherwise, the center is moved to the translation "
              "vector."}});
    docstring::ClassMethodDocInject(
            m, "Geometry3D", "scale",
            {{"scale",
              "The scale parameter that is multiplied to the points/vertices "
              "of the geometry."},
             {"center", "Scale center used for transformation."}});
    docstring::ClassMethodDocInject(
            m, "Geometry3D", "rotate",
            {{"R", "The rotation matrix"},
             {"center", "Rotation center used for transformation."}});

    // open3d.geometry.Geometry2D
    py::class_<Geometry2D, PyGeometry2D<Geometry2D>,
               std::shared_ptr<Geometry2D>, Geometry>
            geometry2d(m, "Geometry2D",
                       "The base geometry class for 2D geometries.");
    geometry2d
            .def("get_min_bound", &Geometry2D::GetMinBound,
                 "Returns min bounds for geometry coordinates.")
            .def("get_max_bound", &Geometry2D::GetMaxBound,
                 "Returns max bounds for geometry coordinates.");
    docstring::ClassMethodDocInject(m, "Geometry2D", "get_min_bound");
    docstring::ClassMethodDocInject(m, "Geometry2D", "get_max_bound");
}

void pybind_geometry(py::module &m) {
    py::module m_submodule = m.def_submodule("geometry");
    pybind_geometry_classes(m_submodule);
    pybind_kdtreeflann(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_keypoint(m_submodule);
    pybind_voxelgrid(m_submodule);
    pybind_lineset(m_submodule);
    pybind_meshbase(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_halfedgetrianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_tetramesh(m_submodule);
    pybind_pointcloud_methods(m_submodule);
    pybind_voxelgrid_methods(m_submodule);
    pybind_meshbase_methods(m_submodule);
    pybind_lineset_methods(m_submodule);
    pybind_image_methods(m_submodule);
    pybind_octree_methods(m_submodule);
    pybind_octree(m_submodule);
    pybind_boundingvolume(m_submodule);
}

}  // namespace geometry
}  // namespace open3d
