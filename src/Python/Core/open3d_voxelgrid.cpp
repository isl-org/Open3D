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

#include "open3d_core.h"
#include "open3d_core_trampoline.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/VoxelGrid.h>
#include <IO/ClassIO/VoxelGridIO.h>
using namespace open3d;

void pybind_voxelgrid(py::module &m) {
    py::class_<VoxelGrid, PyGeometry3D<VoxelGrid>, std::shared_ptr<VoxelGrid>,
               Geometry3D>
            voxelgrid(m, "VoxelGrid");
    py::detail::bind_default_constructor<VoxelGrid>(voxelgrid);
    py::detail::bind_copy_functions<VoxelGrid>(voxelgrid);
    voxelgrid
            .def("__repr__",
                 [](const VoxelGrid &voxelgrid) {
                     return std::string("VoxelGrid with ") +
                            std::to_string(voxelgrid.voxels_.size()) +
                            " voxels.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_voxels", &VoxelGrid::HasVoxels)
            .def("has_colors", &VoxelGrid::HasColors)
            .def_readwrite("voxels", &VoxelGrid::voxels_)
            .def_readwrite("colors", &VoxelGrid::colors_)
            .def_readwrite("origin", &VoxelGrid::origin_)
            .def_readwrite("voxel_size", &VoxelGrid::voxel_size_);
}

void pybind_voxelgrid_methods(py::module &m) {
    m.def("create_surface_voxel_grid_from_point_cloud",
          &CreateSurfaceVoxelGridFromPointCloud,
          "Function to make voxels from scanned point cloud", "point_cloud"_a,
          "voxel_size"_a);
    m.def("read_voxel_grid",
          [](const std::string &filename, const std::string &format) {
              VoxelGrid voxel_grid;
              ReadVoxelGrid(filename, voxel_grid, format);
              return voxel_grid;
          },
          "Function to read VoxelGrid from file", "filename"_a,
          "format"_a = "auto");
    m.def("write_voxel_grid",
          [](const std::string &filename, const VoxelGrid &voxel_grid,
             bool write_ascii, bool compressed) {
              return WriteVoxelGrid(filename, voxel_grid, write_ascii,
                                    compressed);
          },
          "Function to write VoxelGrid to file", "filename"_a, "voxel_grid"_a,
          "write_ascii"_a = false, "compressed"_a = false);
}
