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

#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Python/docstring.h"
#include "Python/geometry/geometry.h"
#include "Python/geometry/geometry_trampoline.h"

using namespace open3d;

void pybind_voxelgrid(py::module &m) {
    py::class_<geometry::VoxelGrid, PyGeometry3D<geometry::VoxelGrid>,
               std::shared_ptr<geometry::VoxelGrid>, geometry::Geometry3D>
            voxelgrid(m, "VoxelGrid",
                      "VoxelGrid is a collection of voxels which are aligned "
                      "in grid.");
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
            .def("has_voxels", &geometry::VoxelGrid::HasVoxels,
                 "Returns ``True`` if the voxel grid contains voxels.")
            .def("has_colors", &geometry::VoxelGrid::HasColors,
                 "Returns ``True`` if the voxel grid contains voxel colors.")
            .def("get_voxel", &geometry::VoxelGrid::GetVoxel, "point"_a,
                 "Returns voxel index given query point.")
            .def_readwrite("voxels", &geometry::VoxelGrid::voxels_,
                           "``int`` array of shape ``(num_voxels, 3)``: "
                           "Voxel coordinates. use ``numpy.asarray()`` to "
                           "access data.")
            .def_readwrite(
                    "colors", &geometry::VoxelGrid::colors_,
                    "``float64`` array of shape ``(num_voxels, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of voxels.")
            .def_readwrite("origin", &geometry::VoxelGrid::origin_,
                           "``float64`` vector of length 3: Coorindate of the "
                           "origin point.")
            .def_readwrite("voxel_size", &geometry::VoxelGrid::voxel_size_);
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_colors");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_voxels");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "get_voxel",
                                    {{"point", "The query point."}});
}

void pybind_voxelgrid_methods(py::module &m) {
    m.def("create_surface_voxel_grid_from_point_cloud",
          &geometry::CreateSurfaceVoxelGridFromPointCloud,
          "Function to make voxels from scanned point cloud", "point_cloud"_a,
          "voxel_size"_a, "min_bound"_a, "max_bound"_a);
    docstring::FunctionDocInject(
            m, "create_surface_voxel_grid_from_point_cloud",
            {{"point_cloud", "The input point cloud."},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."}});
}
