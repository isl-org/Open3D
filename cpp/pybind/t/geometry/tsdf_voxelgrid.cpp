// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <string>
#include <unordered_map>

#include "open3d/t/geometry/TSDFVoxelGrid.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_tsdf_voxelgrid(py::module& m) {
    py::class_<TSDFVoxelGrid> tsdf_voxelgrid(
            m, "TSDFVoxelGrid",
            "A voxel grid for TSDF and/or color integration.");

    // Constructors.
    tsdf_voxelgrid.def(
            py::init<const std::unordered_map<std::string, core::Dtype>&, float,
                     float, int64_t, int64_t, const core::Device&>(),
            "map_attrs_to_dtypes"_a =
                    std::unordered_map<std::string, core::Dtype>{
                            {"tsdf", core::Dtype::Float32},
                            {"weight", core::Dtype::UInt16},
                            {"color", core::Dtype::UInt16},
                    },
            "voxel_size"_a = 3.0 / 512, "sdf_trunc"_a = 0.04,
            "block_resolution"_a = 16, "block_count"_a = 100,
            "device"_a = core::Device("CPU:0"));

    tsdf_voxelgrid.def("integrate",
                       py::overload_cast<const Image&, const core::Tensor&,
                                         const core::Tensor&, float, float>(
                               &TSDFVoxelGrid::Integrate));
    tsdf_voxelgrid.def(
            "integrate",
            py::overload_cast<const Image&, const Image&, const core::Tensor&,
                              const core::Tensor&, float, float>(
                    &TSDFVoxelGrid::Integrate));

    tsdf_voxelgrid.def("extract_surface_points",
                       &TSDFVoxelGrid::ExtractSurfacePoints);
    tsdf_voxelgrid.def("extract_surface_mesh",
                       &TSDFVoxelGrid::ExtractSurfaceMesh);

    tsdf_voxelgrid.def("to", &TSDFVoxelGrid::To, "device"_a, "copy"_a = false);
    tsdf_voxelgrid.def("clone", &TSDFVoxelGrid::Clone);
    tsdf_voxelgrid.def("cpu", &TSDFVoxelGrid::CPU);
    tsdf_voxelgrid.def("cuda", &TSDFVoxelGrid::CUDA, "device_id"_a);

    tsdf_voxelgrid.def("get_device", &TSDFVoxelGrid::GetDevice);
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
