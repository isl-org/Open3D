// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace geometry {

void pybind_geometry(py::module &m);

void pybind_pointcloud(py::module &m);
void pybind_keypoint(py::module &m);
void pybind_voxelgrid(py::module &m);
void pybind_lineset(py::module &m);
void pybind_meshbase(py::module &m);
void pybind_trianglemesh(py::module &m);
void pybind_halfedgetrianglemesh(py::module &m);
void pybind_image(py::module &m);
void pybind_tetramesh(py::module &m);
void pybind_kdtreeflann(py::module &m);
void pybind_pointcloud_methods(py::module &m);
void pybind_voxelgrid_methods(py::module &m);
void pybind_meshbase_methods(py::module &m);
void pybind_trianglemesh_methods(py::module &m);
void pybind_lineset_methods(py::module &m);
void pybind_image_methods(py::module &m);
void pybind_octree_methods(py::module &m);
void pybind_octree(py::module &m);
void pybind_boundingvolume(py::module &m);

}  // namespace geometry
}  // namespace open3d
