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

#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

#include <Open3D/Open3D.h>
using namespace open3d;

void pybind_geometry_classes(py::module &m) {
    py::class_<geometry::Geometry, PyGeometry<geometry::Geometry>,
               std::shared_ptr<geometry::Geometry>>
            geometry(m, "Geometry", "Geometry");
    geometry.def("clear", &geometry::Geometry::Clear)
            .def("is_empty", &geometry::Geometry::IsEmpty)
            .def("get_geometry_type", &geometry::Geometry::GetGeometryType)
            .def("dimension", &geometry::Geometry::Dimension);
    py::enum_<geometry::Geometry::GeometryType>(geometry, "Type",
                                                py::arithmetic())
            .value("Unspecified", geometry::Geometry::GeometryType::Unspecified)
            .value("PointCloud", geometry::Geometry::GeometryType::PointCloud)
            .value("VoxelGrid", geometry::Geometry::GeometryType::VoxelGrid)
            .value("LineSet", geometry::Geometry::GeometryType::LineSet)
            .value("TriangleMesh",
                   geometry::Geometry::GeometryType::TriangleMesh)
            .value("HalfEdgeTriangleMesh",
                   geometry::Geometry::GeometryType::HalfEdgeTriangleMesh)
            .value("Image", geometry::Geometry::GeometryType::Image)
            .export_values();

    py::class_<geometry::Geometry3D, PyGeometry3D<geometry::Geometry3D>,
               std::shared_ptr<geometry::Geometry3D>, geometry::Geometry>
            geometry3d(m, "Geometry3D", "Geometry3D");
    geometry3d.def("get_min_bound", &geometry::Geometry3D::GetMinBound)
            .def("get_max_bound", &geometry::Geometry3D::GetMaxBound)
            .def("transform", &geometry::Geometry3D::Transform);

    py::class_<geometry::Geometry2D, PyGeometry2D<geometry::Geometry2D>,
               std::shared_ptr<geometry::Geometry2D>, geometry::Geometry>
            geometry2d(m, "Geometry2D", "Geometry2D");
    geometry2d.def("get_min_bound", &geometry::Geometry2D::GetMinBound)
            .def("get_max_bound", &geometry::Geometry2D::GetMaxBound);
}

void pybind_geometry(py::module &m) {
    py::module m_submodule = m.def_submodule("geometry");
    pybind_geometry_classes(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_voxelgrid(m_submodule);
    pybind_lineset(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_halfedgetrianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_kdtreeflann(m_submodule);
    pybind_pointcloud_methods(m_submodule);
    pybind_voxelgrid_methods(m_submodule);
    pybind_trianglemesh_methods(m_submodule);
    pybind_image_methods(m_submodule);
}
