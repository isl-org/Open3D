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

#include <string>
#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/VoxelBlockGrid.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_voxel_block_grid(py::module& m) {
    py::class_<VoxelBlockGrid> vbg(
            m, "VoxelBlockGrid",
            "A voxel block grid is a sparse grid of voxel blocks. Each voxel "
            "block is a dense 3D array, preserving local data distribution. If "
            "the block_resolution is set to 1, then the VoxelBlockGrid "
            "degenerates to a sparse voxel grid.");

    vbg.def(py::init([](const std::vector<std::string>& attr_names,
                        const std::vector<core::Dtype>& attr_dtypes,
                        const std::vector<py::handle>& attr_channels,
                        float voxel_size, int64_t block_resolution,
                        int64_t block_count, const core::Device& device) {
                std::vector<core::SizeVector> attr_channel_svs;
                for (const auto& handle : attr_channels) {
                    auto attr_channel = core::PyHandleToSizeVector(handle);
                    attr_channel_svs.push_back(attr_channel);
                }

                return VoxelBlockGrid(attr_names, attr_dtypes, attr_channel_svs,
                                      voxel_size, block_resolution, block_count,
                                      device);
            }),
            "attr_names"_a, "attr_dtypes"_a, "attr_channels"_a,
            "voxel_size"_a = 0.0058, "block_resolution"_a = 8,
            "block_count"_a = 10000, "device"_a = core::Device("CPU:0"));

    vbg.def("hashmap", &VoxelBlockGrid::GetHashMap,
            "Get the underlying hash map from 3d block coordinates to block "
            "voxel grids.");

    vbg.def("attribute", &VoxelBlockGrid::GetAttribute,
            "Get the attribute tensor to be indexed with voxel_indices.",
            "attribute_name"_a);

    vbg.def("voxel_indices", &VoxelBlockGrid::GetVoxelIndices,
            "Get a (4, N), Int64 index tensor for active voxels, used for "
            "advanced indexing.                                                "
            "Returned index tensor can access selected value buffers in order "
            "of  "
            "(buf_index, index_voxel_x, index_voxel_y, index_voxel_z).         "
            "Example:                                                          "
            "For a voxel block grid with (2, 2, 2) block resolution,           "
            "if the active block coordinates are at buffer index {(2, 4)} "
            "given by"
            "active_indices() from the underlying hash map,                    "
            "the returned result will be a (4, 2 x 8) tensor:                  "
            "{                                                                 "
            "(2, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 1, 1, 0),           "
            "(2, 0, 0, 1), (2, 1, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1),           "
            "(4, 0, 0, 0), (4, 1, 0, 0), (4, 0, 1, 0), (4, 1, 1, 0),           "
            "(4, 0, 0, 1), (4, 1, 0, 1), (4, 0, 1, 1), (4, 1, 1, 1),           "
            "}"
            "Note: the slicing order is z-y-x.");

    vbg.def("voxel_coordinates", &VoxelBlockGrid::GetVoxelCoordinates,
            "Get a (3, hashmap.Size() * resolution^3) coordinate tensor of "
            "active"
            "voxels per block, used for geometry transformation jointly with   "
            "indices from voxel_indices.                                       "
            "Example:                                                          "
            "For a voxel block grid with (2, 2, 2) block resolution,           "
            "if the active block coordinates are {(-1, 3, 2), (0, 2, 4)},      "
            "the returned result will be a (3, 2 x 8) tensor given by:         "
            "{                                                                 "
            "key_tensor[voxel_indices[0]] * block_resolution_ + "
            "voxel_indices[1] "
            "key_tensor[voxel_indices[0]] * block_resolution_ + "
            "voxel_indices[2] "
            "key_tensor[voxel_indices[0]] * block_resolution_ + "
            "voxel_indices[3] "
            "}                                                                 "
            "Note: the coordinates are VOXEL COORDINATES in Int64. To access "
            "metric"
            "coordinates, multiply by voxel size.",
            "voxel_indices"_a);

    vbg.def("compute_unique_block_coordinates",
            py::overload_cast<const Image&, const core::Tensor&,
                              const core::Tensor&, float, float>(
                    &VoxelBlockGrid::GetUniqueBlockCoordinates),
            "Get a (3, M) active block coordinates from a depth image, with "
            "potential duplicates removed."
            "Note: these coordinates are not activated in the internal sparse "
            "voxel block. They need to be inserted in the hash map.",
            "depth"_a, "intrinsic"_a, "extrinsic"_a, "depth_scale"_a = 1000.0f,
            "depth_max"_a = 3.0f);

    vbg.def("compute_unique_block_coordinates",
            py::overload_cast<const PointCloud&>(
                    &VoxelBlockGrid::GetUniqueBlockCoordinates),
            "Obtain active block coordinates from a point cloud.", "pcd"_a);

    vbg.def("integrate", &VoxelBlockGrid::Integrate,
            "Specific operation for TSDF volumes."
            "Integrate an RGB-D frame in the selected block coordinates using "
            "pinhole camera model.",
            "block_coords"_a, "depth"_a, "color"_a, "intrinsic"_a,
            "extrinsic"_a, "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f);

    vbg.def("ray_cast", &VoxelBlockGrid::RayCast,
            "Specific operation for TSDF volumes."
            "Perform volumetric ray casting in the selected block coordinates."
            "The block coordinates in the frustum can be taken from"
            "compute_unique_block_coordinates"
            "All the block coordinates can be taken from "
            "hashmap().key_tensor()",
            "block_coords"_a, "intrinsic"_a, "extrinsic"_a, "width"_a,
            "height"_a, "depth_scale"_a = 1000.0f, "depth_min"_a = 0.1f,
            "depth_max"_a = 3.0f, "weight_threshold"_a = 3.0f);

    vbg.def("extract_point_cloud", &VoxelBlockGrid::ExtractPointCloud,
            "Specific operation for TSDF volumes."
            "Extract point cloud at isosurface points.",
            "point_cloud_size_estimate"_a = -1, "weight_threshold"_a = 3.0f);

    vbg.def("extract_triangle_mesh", &VoxelBlockGrid::ExtractTriangleMesh,
            "Specific operation for TSDF volumes."
            "Extract triangle mesh at isosurface points.",
            "vertex_size_estimate"_a = -1, "weight_threshold"_a = 3.0f);

    vbg.def("save", &VoxelBlockGrid::Save,
            "Save the voxel block grid to a npz file."
            "file_name"_a);
    vbg.def_static("load", &VoxelBlockGrid::Load,
                   "Load a voxel block grid from a npz file.", "file_name"_a);
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
