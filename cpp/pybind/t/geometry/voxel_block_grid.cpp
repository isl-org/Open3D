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

    vbg.def(py::init<const std::vector<std::string>&,
                     const std::vector<core::Dtype>&,
                     const std::vector<core::SizeVector>&, float, int64_t,
                     int64_t, const core::Device&>(),
            "attr_names"_a, "attr_dtypes"_a, "attr_channels"_a,
            "voxel_size"_a = 0.0058, "block_resolution"_a = 16,
            "block_count"_a = 10000, "device"_a = core::Device("CPU:0"));

    vbg.def("hashmap", &VoxelBlockGrid::GetHashMap,
            "Get the underlying hash map from 3d block coordinates to block "
            "voxel grids.");

    vbg.def("attribute", &VoxelBlockGrid::GetAttribute,
            "Get the attribute tensor to be indexed with voxel_indices.",
            "attribute_name"_a);

    vbg.def("voxel_indices",
            py::overload_cast<const core::Tensor&>(
                    &VoxelBlockGrid::GetVoxelIndices, py::const_),
            "Get a (4, N), Int64 index tensor for input buffer indices, used "
            "for advanced indexing.   "
            "Returned index tensor can access selected value buffer"
            "in the order of  "
            "(buf_index, index_voxel_x, index_voxel_y, index_voxel_z).       "
            "Example:                                                        "
            "For a voxel block grid with (2, 2, 2) block resolution,         "
            "if the active block coordinates are at buffer index {(2, 4)} "
            "given by active_indices() from the underlying hash map,         "
            "the returned result will be a (4, 2 x 8) tensor:                "
            "{                                                               "
            "(2, 0, 0, 0), (2, 1, 0, 0), (2, 0, 1, 0), (2, 1, 1, 0),         "
            "(2, 0, 0, 1), (2, 1, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1),         "
            "(4, 0, 0, 0), (4, 1, 0, 0), (4, 0, 1, 0), (4, 1, 1, 0),         "
            "(4, 0, 0, 1), (4, 1, 0, 1), (4, 0, 1, 1), (4, 1, 1, 1),         "
            "}"
            "Note: the slicing order is z-y-x.");

    vbg.def("voxel_indices",
            py::overload_cast<>(&VoxelBlockGrid::GetVoxelIndices, py::const_),
            "Get a (4, N) Int64 idnex tensor for all the active voxels stored "
            "in the hash map, used for advanced indexing.");

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

    vbg.def("voxel_coordinates_and_flattened_indices",
            py::overload_cast<const core::Tensor&>(
                    &VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices),
            "Get a (buf_indices.shape[0] * resolution^3, 3), Float32 voxel "
            "coordinate tensor,"
            "and a (buf_indices.shape[0] * resolution^3, 1), Int64 voxel index "
            "tensor.",
            "buf_indices"_a);

    vbg.def("voxel_coordinates_and_flattened_indices",
            py::overload_cast<>(
                    &VoxelBlockGrid::GetVoxelCoordinatesAndFlattenedIndices),
            "Get a (hashmap.size() * resolution^3, 3), Float32 voxel "
            "coordinate tensor,"
            "and a (hashmap.size() * resolution^3, 1), Int64 voxel index "
            "tensor.");

    vbg.def("compute_unique_block_coordinates",
            py::overload_cast<const Image&, const core::Tensor&,
                              const core::Tensor&, float, float, float>(
                    &VoxelBlockGrid::GetUniqueBlockCoordinates),
            "Get a (3, M) active block coordinates from a depth image, with "
            "potential duplicates removed."
            "Note: these coordinates are not activated in the internal sparse "
            "voxel block. They need to be inserted in the hash map.",
            "depth"_a, "intrinsic"_a, "extrinsic"_a, "depth_scale"_a = 1000.0f,
            "depth_max"_a = 3.0f, "trunc_voxel_multiplier"_a = 8.0);

    vbg.def("compute_unique_block_coordinates",
            py::overload_cast<const PointCloud&, float>(
                    &VoxelBlockGrid::GetUniqueBlockCoordinates),
            "Obtain active block coordinates from a point cloud.", "pcd"_a,
            "trunc_voxel_multiplier"_a = 8.0);

    vbg.def("integrate",
            py::overload_cast<const core::Tensor&, const Image&, const Image&,
                              const core::Tensor&, const core::Tensor&,
                              const core::Tensor&, float, float, float>(
                    &VoxelBlockGrid::Integrate),
            "Specific operation for TSDF volumes."
            "Integrate an RGB-D frame in the selected block coordinates using "
            "pinhole camera model.",
            "block_coords"_a, "depth"_a, "color"_a, "depth_intrinsic"_a,
            "color_intrinsic"_a, "extrinsic"_a,
            "depth_scale"_a.noconvert() = 1000.0f,
            "depth_max"_a.noconvert() = 3.0f,
            "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

    vbg.def("integrate",
            py::overload_cast<const core::Tensor&, const Image&, const Image&,
                              const core::Tensor&, const core::Tensor&, float,
                              float, float>(&VoxelBlockGrid::Integrate),
            "Specific operation for TSDF volumes."
            "Integrate an RGB-D frame in the selected block coordinates using "
            "pinhole camera model.",
            "block_coords"_a, "depth"_a, "color"_a, "intrinsic"_a,
            "extrinsic"_a, "depth_scale"_a.noconvert() = 1000.0f,
            "depth_max"_a.noconvert() = 3.0f,
            "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

    vbg.def("integrate",
            py::overload_cast<const core::Tensor&, const Image&,
                              const core::Tensor&, const core::Tensor&, float,
                              float, float>(&VoxelBlockGrid::Integrate),
            "Specific operation for TSDF volumes."
            "Similar to RGB-D integration, but only applied to depth images.",
            "block_coords"_a, "depth"_a, "intrinsic"_a, "extrinsic"_a,
            "depth_scale"_a.noconvert() = 1000.0f,
            "depth_max"_a.noconvert() = 3.0f,
            "trunc_voxel_multiplier"_a.noconvert() = 8.0f);

    vbg.def("ray_cast", &VoxelBlockGrid::RayCast,
            "Specific operation for TSDF volumes."
            "Perform volumetric ray casting in the selected block coordinates."
            "The block coordinates in the frustum can be taken from"
            "compute_unique_block_coordinates"
            "All the block coordinates can be taken from "
            "hashmap().key_tensor()",
            "block_coords"_a, "intrinsic"_a, "extrinsic"_a, "width"_a,
            "height"_a,
            "render_attributes"_a = std::vector<std::string>{"depth", "color"},
            "depth_scale"_a = 1000.0f, "depth_min"_a = 0.1f,
            "depth_max"_a = 3.0f, "weight_threshold"_a = 3.0f,
            "trunc_voxel_multiplier"_a = 8.0f, "range_map_down_factor"_a = 8);

    vbg.def("extract_point_cloud", &VoxelBlockGrid::ExtractPointCloud,
            "Specific operation for TSDF volumes."
            "Extract point cloud at isosurface points.",
            "weight_threshold"_a = 3.0f, "estimated_point_number"_a = -1);

    vbg.def("extract_triangle_mesh", &VoxelBlockGrid::ExtractTriangleMesh,
            "Specific operation for TSDF volumes."
            "Extract triangle mesh at isosurface points.",
            "weight_threshold"_a = 3.0f, "estimated_vertex_number"_a = -1);

    // Device transfers.
    vbg.def("to", &VoxelBlockGrid::To,
            "Transfer the voxel block grid to a specified device.", "device"_a,
            "copy"_a = false);

    vbg.def(
            "cpu",
            [](const VoxelBlockGrid& voxelBlockGrid) {
                return voxelBlockGrid.To(core::Device("CPU:0"));
            },
            "Transfer the voxel block grid to CPU. If the voxel block grid is "
            "already on CPU, no copy will be performed.");
    vbg.def(
            "cuda",
            [](const VoxelBlockGrid& voxelBlockGrid, int device_id) {
                return voxelBlockGrid.To(core::Device("CUDA", device_id));
            },
            "Transfer the voxel block grid to a CUDA device. If the voxel "
            "block grid is already on the specified CUDA device, no copy "
            "will be performed.",
            "device_id"_a = 0);

    vbg.def("save", &VoxelBlockGrid::Save,
            "Save the voxel block grid to a npz file."
            "file_name"_a);
    vbg.def_static("load", &VoxelBlockGrid::Load,
                   "Load a voxel block grid from a npz file.", "file_name"_a);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
