// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace geometry {

void pybind_geometry_declarations(py::module &m);
void pybind_kdtreeflann_declarations(py::module &m);
void pybind_pointcloud_declarations(py::module &m);
void pybind_keypoint_declarations(py::module &m);
void pybind_voxelgrid_declarations(py::module &m);
void pybind_lineset_declarations(py::module &m);
void pybind_meshbase_declarations(py::module &m);
void pybind_trianglemesh_declarations(py::module &m);
void pybind_halfedgetrianglemesh_declarations(py::module &m);
void pybind_image_declarations(py::module &m);
void pybind_tetramesh_declarations(py::module &m);
void pybind_octree_declarations(py::module &m);
void pybind_boundingvolume_declarations(py::module &m);

void pybind_geometry_definitions(py::module &m);
void pybind_kdtreeflann_definitions(py::module &m);
void pybind_pointcloud_definitions(py::module &m);
void pybind_keypoint_definitions(py::module &m);
void pybind_voxelgrid_definitions(py::module &m);
void pybind_lineset_definitions(py::module &m);
void pybind_meshbase_definitions(py::module &m);
void pybind_trianglemesh_definitions(py::module &m);
void pybind_halfedgetrianglemesh_definitions(py::module &m);
void pybind_image_definitions(py::module &m);
void pybind_tetramesh_definitions(py::module &m);
void pybind_octree_definitions(py::module &m);
void pybind_boundingvolume_definitions(py::module &m);

}  // namespace geometry
}  // namespace open3d
