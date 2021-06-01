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
    py::enum_<TSDFVoxelGrid::SurfaceMaskCode>(
            m, "SurfaceMaskCode",
            "Mask code for surface extraction used in raycasting and surface "
            "extraction.")
            .value("VertexMap", TSDFVoxelGrid::SurfaceMaskCode::VertexMap)
            .value("DepthMap", TSDFVoxelGrid::SurfaceMaskCode::DepthMap)
            .value("ColorMap", TSDFVoxelGrid::SurfaceMaskCode::ColorMap)
            .value("NormalMap", TSDFVoxelGrid::SurfaceMaskCode::NormalMap)
            .export_values();

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
                               &TSDFVoxelGrid::Integrate),
                       "depth"_a, "intrinsics"_a, "extrinsics"_a,
                       "depth_scale"_a, "depth_max"_a);

    tsdf_voxelgrid.def(
            "integrate",
            py::overload_cast<const Image&, const Image&, const core::Tensor&,
                              const core::Tensor&, float, float>(
                    &TSDFVoxelGrid::Integrate),
            "depth"_a, "color"_a, "intrinsics"_a, "extrinsics"_a,
            "depth_scale"_a, "depth_max"_a);

    // TODO(wei): expose mask code as a python class
    tsdf_voxelgrid.def(
            "raycast", &TSDFVoxelGrid::RayCast, "intrinsics"_a, "extrinsics"_a,
            "width"_a, "height"_a, "depth_scale"_a = 1000.0,
            "depth_min"_a = 0.1f, "depth_max"_a = 3.0f,
            "weight_threshold"_a = 3.0f,
            "raycast_result_mask"_a = TSDFVoxelGrid::SurfaceMaskCode::DepthMap |
                                      TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
    tsdf_voxelgrid.def(
            "extract_surface_points", &TSDFVoxelGrid::ExtractSurfacePoints,
            "estimate_number"_a = -1, "weight_threshold"_a = 3.0f,
            "surface_mask"_a = TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
                               TSDFVoxelGrid::SurfaceMaskCode::ColorMap);
    tsdf_voxelgrid.def(
            "extract_surface_mesh", &TSDFVoxelGrid::ExtractSurfaceMesh,
            "estimate_number"_a = -1, "weight_threshold"_a = 3.0f,
            "surface_mask"_a = TSDFVoxelGrid::SurfaceMaskCode::VertexMap |
                               TSDFVoxelGrid::SurfaceMaskCode::ColorMap |
                               TSDFVoxelGrid::SurfaceMaskCode::NormalMap);

    tsdf_voxelgrid.def("to", &TSDFVoxelGrid::To, "device"_a, "copy"_a = false);
    tsdf_voxelgrid.def("clone", &TSDFVoxelGrid::Clone);
    tsdf_voxelgrid.def("cpu", &TSDFVoxelGrid::CPU);
    tsdf_voxelgrid.def("cuda", &TSDFVoxelGrid::CUDA, "device_id"_a);

    tsdf_voxelgrid.def("get_block_hashmap", &TSDFVoxelGrid::GetBlockHashmap);
    tsdf_voxelgrid.def("get_device", &TSDFVoxelGrid::GetDevice);
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
