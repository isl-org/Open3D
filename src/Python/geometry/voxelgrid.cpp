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

#include "Python/geometry/geometry_trampoline.h"
#include "Python/geometry/geometry.h"

#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/VoxelGrid.h>

using namespace open3d;

void pybind_voxelgrid(py::module &m) {
    py::class_<geometry::VoxelGrid, PyGeometry3D<geometry::VoxelGrid>,
               std::shared_ptr<geometry::VoxelGrid>, geometry::Geometry3D>
            voxelgrid(m, "VoxelGrid");
    py::detail::bind_default_constructor<geometry::VoxelGrid>(voxelgrid);
    py::detail::bind_copy_functions<geometry::VoxelGrid>(voxelgrid);
    voxelgrid
            .def("__repr__",
                 [](const geometry::VoxelGrid &voxelgrid) {
                     return std::string("geometry::VoxelGrid with ") +
                            std::to_string(voxelgrid.voxels_.size()) +
                            " voxels.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_voxels", &geometry::VoxelGrid::HasVoxels)
            .def("has_colors", &geometry::VoxelGrid::HasColors)
            .def_readwrite("voxels", &geometry::VoxelGrid::voxels_.h_data)
            .def_readwrite("colors", &geometry::VoxelGrid::colors_.h_data)
            .def_readwrite("origin", &geometry::VoxelGrid::origin_)
            .def_readwrite("voxel_size", &geometry::VoxelGrid::voxel_size_);
}

void pybind_voxelgrid_methods(py::module &m) {
    m.def("create_surface_voxel_grid_from_point_cloud",
          &geometry::CreateSurfaceVoxelGridFromPointCloud,
          "Function to make voxels from scanned point cloud", "point_cloud"_a,
          "voxel_size"_a);
}
