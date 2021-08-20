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

#include "open3d/t/geometry/VoxelBlockGrid.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"

namespace open3d {
namespace t {
namespace geometry {

VoxelBlockGrid::VoxelBlockGrid(
        const std::vector<std::string> &attr_names,
        const std::vector<core::Dtype> &attr_dtypes,
        const std::vector<core::SizeVector> &attr_channels,
        double voxel_size,
        int64_t block_resolution,
        int64_t block_count,
        const core::Device &device,
        const core::HashBackendType &backend)
    : voxel_size_(voxel_size), block_resolution_(block_resolution) {
    size_t n_attrs = attr_names.size();
    if (attr_dtypes.size() != n_attrs) {
        utility::LogError(
                "Number of attribute dtypes ({}) mismatch with names ({}).",
                attr_dtypes.size(), n_attrs);
    }
    if (attr_channels.size() != n_attrs) {
        utility::LogError(
                "Number of attribute channels ({}) mismatch with names ({}).",
                attr_channels.size(), n_attrs);
    }

    std::vector<core::SizeVector> attr_element_shapes;
    core::SizeVector block_shape{block_resolution, block_resolution,
                                 block_resolution};

    for (size_t i = 0; i < n_attrs; ++i) {
        // Construct element shapes.
        core::SizeVector attr_channel = attr_channels[i];
        core::SizeVector block_shape_copy = block_shape;
        block_shape_copy.insert(block_shape_copy.end(), attr_channel.begin(),
                                attr_channel.end());
        attr_element_shapes.emplace_back(block_shape_copy);

        // Used for easier accessing via attribute names.
        name_attr_map_[attr_names[i]] = i;
    }

    block_hashmap_ = std::make_shared<core::HashMap>(
            block_count, core::Int32, core::SizeVector{3}, attr_dtypes,
            attr_element_shapes, device, backend);
}

void VoxelBlockGrid::Integrate(const Image &depth,
                               const Image &color,
                               const core::Tensor &intrinsics,
                               const core::Tensor &extrinsics,
                               float depth_scale,
                               float depth_max) {
    if (depth.IsEmpty()) {
        utility::LogError("Input depth is empty.");
    }
    if (depth.GetDtype() != core::UInt16 && depth.GetDtype() != core::Float32) {
        utility::LogError("Unsupported depth image dtype {}.",
                          depth.GetDtype().ToString());
    }

    const int64_t down_factor = 4;
    if (point_hashmap_ == nullptr) {
        int64_t capacity = (depth.GetCols() * depth.GetRows()) /
                           (down_factor * down_factor * 4);
        point_hashmap_ = std::make_shared<core::HashMap>(
                capacity, core::Int32, core::SizeVector{3}, core::Int32,
                core::SizeVector{1}, block_hashmap_->GetDevice());
    } else {
        point_hashmap_->Clear();
    }

    core::Tensor block_coords;
    PointCloud pcd = PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics, depth_scale, depth_max, down_factor);
    kernel::tsdf::Touch(point_hashmap_, pcd.GetPointPositions().Contiguous(),
                        block_coords, block_resolution_, voxel_size_,
                        6 * voxel_size_);

    utility::LogInfo("block_coords's shape = {}", block_coords.GetShape());

    // Active voxel blocks in the block hashmap.
    core::Tensor buf_indices, masks;
    int64_t n = block_hashmap_->Size();
    try {
        block_hashmap_->Activate(block_coords, buf_indices, masks);
    } catch (const std::runtime_error &) {
        utility::LogError(
                "[TSDFIntegrate] Unable to allocate volume during rehashing. "
                "Consider using a "
                "larger block_count at initialization to avoid rehashing "
                "(currently {}), or choosing a larger voxel_size "
                "(currently {})",
                n, voxel_size_);
    }
    block_hashmap_->Find(block_coords, buf_indices, masks);

    core::Tensor block_keys = block_hashmap_->GetKeyTensor();
    std::vector<core::Tensor> block_values = block_hashmap_->GetValueTensors();
    kernel::tsdf::Integrate(depth.AsTensor(), color.AsTensor(), buf_indices,
                            block_keys, block_values, intrinsics, extrinsics,
                            block_resolution_, voxel_size_, voxel_size_ * 6,
                            depth_scale, depth_max);
}

std::pair<core::Tensor, core::Tensor> BufferRadiusNeighbors(
        std::shared_ptr<core::HashMap> &hashmap,
        const core::Tensor &active_buf_indices) {
    // Fixed radius search for spatially hashed voxel blocks.
    // A generalization will be implementing dense/sparse fixed radius search
    // with coordinates as hashmap keys.
    core::Tensor key_buffer_int3_tensor = hashmap->GetKeyTensor();

    core::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
            {active_buf_indices.To(core::Int64)});
    int64_t n = active_keys.GetShape()[0];

    // Fill in radius nearest neighbors.
    core::Tensor keys_nb({27, n, 3}, core::Int32, hashmap->GetDevice());
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        core::Tensor dt =
                core::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1}, {1, 3},
                             core::Int32, hashmap->GetDevice());
        keys_nb[nb] = active_keys + dt;
    }
    keys_nb = keys_nb.View({27 * n, 3});

    core::Tensor buf_indices_nb, masks_nb;
    hashmap->Find(keys_nb, buf_indices_nb, masks_nb);
    return std::make_pair(buf_indices_nb.View({27, n, 1}),
                          masks_nb.View({27, n, 1}));
}

PointCloud VoxelBlockGrid::ExtractSurfacePoints(int estimated_number,
                                                float weight_threshold) {
    core::Tensor active_buf_indices;
    block_hashmap_->GetActiveIndices(active_buf_indices);

    core::Tensor active_nb_buf_indices, active_nb_masks;
    std::tie(active_nb_buf_indices, active_nb_masks) =
            BufferRadiusNeighbors(block_hashmap_, active_buf_indices);

    // Extract points around zero-crossings.
    core::Tensor points, normals, colors;

    core::Tensor block_keys = block_hashmap_->GetKeyTensor();
    std::vector<core::Tensor> block_values = block_hashmap_->GetValueTensors();
    kernel::tsdf::ExtractSurfacePoints(
            active_buf_indices, active_nb_buf_indices, active_nb_masks,
            block_keys, block_values, points, normals, colors,
            block_resolution_, voxel_size_, weight_threshold, estimated_number);

    auto pcd = PointCloud(points.Slice(0, 0, estimated_number));
    pcd.SetPointColors(colors.Slice(0, 0, estimated_number));
    // pcd.SetPointNormals(normals.Slice(0, 0, estimated_number));

    return pcd;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
