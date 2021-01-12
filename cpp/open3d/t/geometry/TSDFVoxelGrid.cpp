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

#include "open3d/t/geometry/TSDFVoxelGrid.h"

#include "open3d/Open3D.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

TSDFVoxelGrid::TSDFVoxelGrid(
        std::unordered_map<std::string, core::Dtype> attr_dtype_map,
        float voxel_size,
        float sdf_trunc,
        int64_t block_resolution,
        int64_t block_count,
        const core::Device &device)
    : voxel_size_(voxel_size),
      sdf_trunc_(sdf_trunc),
      block_resolution_(block_resolution),
      block_count_(block_count),
      device_(device),
      attr_dtype_map_(attr_dtype_map) {
    if (attr_dtype_map_.count("tsdf") == 0 ||
        attr_dtype_map_.count("weight") == 0) {
        utility::LogError(
                "[TSDFVoxelGrid] expected properties tsdf and weight are "
                "missing.");
    }

    int64_t total_bytes = 0;
    if (attr_dtype_map_.count("tsdf") != 0) {
        core::Dtype dtype = attr_dtype_map_.at("tsdf");
        if (dtype != core::Dtype::Float32) {
            utility::LogWarning(
                    "[TSDFVoxelGrid] unexpected TSDF dtype, please "
                    "implement your own Voxel structure in "
                    "core/kernel/GeneralEWSharedImpl.h for dispatching.");
        }
        total_bytes += dtype.ByteSize();
    }

    if (attr_dtype_map_.count("weight") != 0) {
        core::Dtype dtype = attr_dtype_map_.at("weight");
        if (dtype != core::Dtype::Float32 && dtype != core::Dtype::UInt16) {
            utility::LogWarning(
                    "[TSDFVoxelGrid] unexpected weight dtype, please "
                    "implement your own Voxel structure in "
                    "core/kernel/GeneralEWSharedImpl.h for "
                    "dispatching.");
        }
        total_bytes += dtype.ByteSize();
    }

    if (attr_dtype_map_.count("color") != 0) {
        core::Dtype dtype = attr_dtype_map_.at("color");
        if (dtype != core::Dtype::Float32 && dtype != core::Dtype::UInt16) {
            utility::LogWarning(
                    "[TSDFVoxelGrid] unexpected color dtype, please "
                    "implement your own Voxel structure in "
                    "core/kernel/GeneralEWSharedImpl.h for dispatching.");
        }
        total_bytes += dtype.ByteSize() * 3;
    }
    // Users can add other key/dtype checkers here for potential extensions.

    // SDF trunc check, critical for TSDF touch operation that allocates TSDF
    // volumes.
    if (sdf_trunc > block_resolution_ * voxel_size_ * 0.499) {
        utility::LogError(
                "SDF trunc is too large. Please make sure sdf trunc is smaller "
                "than half block size (i.e., block_resolution * voxel_size * "
                "0.5)");
    }
    block_hashmap_ = std::make_shared<core::Hashmap>(
            block_count_, core::Dtype::Int32, core::Dtype::UInt8,
            core::SizeVector{3},
            core::SizeVector{block_resolution_, block_resolution_,
                             block_resolution_, total_bytes},
            device);
}

void TSDFVoxelGrid::Integrate(const Image &depth,
                              const core::Tensor &intrinsics,
                              const core::Tensor &extrinsics,
                              float depth_scale,
                              float depth_max) {
    Image empty_color;
    Integrate(depth, empty_color, intrinsics, extrinsics, depth_scale,
              depth_max);
}

void TSDFVoxelGrid::Integrate(const Image &depth,
                              const Image &color,
                              const core::Tensor &intrinsics,
                              const core::Tensor &extrinsics,
                              float depth_scale,
                              float depth_max) {
    if (depth.IsEmpty()) {
        utility::LogError(
                "[TSDFVoxelGrid] input depth is empty for integration.");
    }

    // Create a point cloud from a low-resolution depth input to roughly
    // estimate surfaces.
    PointCloud pcd = PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics, depth_scale, depth_max, 4);

    core::Tensor block_coords;
    kernel::tsdf::Touch(pcd.GetPoints().Contiguous(), block_coords,
                        block_resolution_, voxel_size_, sdf_trunc_);

    // Active voxel blocks in the block hashmap.
    core::Tensor addrs, masks;
    int64_t n = block_hashmap_->Size();
    try {
        block_hashmap_->Activate(block_coords, addrs, masks);
    } catch (const std::runtime_error &) {
        utility::LogError(
                "[TSDFIntegrate] Unable to allocate volume during rehashing. "
                "Consider using a "
                "larger block_count at initialization to avoid rehashing "
                "(currently {}), or choosing a larger voxel_size "
                "(currently {})",
                n, voxel_size_);
    }

    // Collect voxel blocks in the viewing frustum. Note we cannot directly
    // reuse addrs from Activate, since some blocks might have been activated in
    // previous launches and return false.
    block_hashmap_->Find(block_coords, addrs, masks);

    core::Tensor depth_tensor = depth.AsTensor().Contiguous();
    core::Tensor color_tensor;
    if (color.IsEmpty()) {
        utility::LogDebug(
                "[TSDFIntegrate] color image is empty, perform depth "
                "integration only.");
    } else if (color.GetRows() == depth.GetRows() &&
               color.GetCols() == depth.GetCols() && color.GetChannels() == 3) {
        if (attr_dtype_map_.count("color") != 0) {
            color_tensor =
                    color.AsTensor().To(core::Dtype::Float32).Contiguous();
        } else {
            utility::LogWarning(
                    "[TSDFIntegrate] color image is ignored since voxels do "
                    "not contain colors.");
        }
    } else {
        utility::LogWarning(
                "[TSDFIntegrate] color image is ignored for the incompatible "
                "shape.");
    }

    core::Tensor dst = block_hashmap_->GetValueTensor();
    kernel::tsdf::Integrate(depth_tensor, color_tensor,
                            addrs.To(core::Dtype::Int64).IndexGet({masks}),
                            block_hashmap_->GetKeyTensor(), dst, intrinsics,
                            extrinsics, block_resolution_, voxel_size_,
                            sdf_trunc_, depth_scale, depth_max);
}

PointCloud TSDFVoxelGrid::ExtractSurfacePoints() {
    // Extract active voxel blocks from the hashmap.
    core::Tensor active_addrs;
    block_hashmap_->GetActiveIndices(active_addrs);
    core::Tensor active_nb_addrs, active_nb_masks;
    std::tie(active_nb_addrs, active_nb_masks) =
            BufferRadiusNeighbors(active_addrs);

    // Extract points around zero-crossings.
    core::Tensor points, normals, colors;
    kernel::tsdf::ExtractSurfacePoints(
            active_addrs.To(core::Dtype::Int64),
            active_nb_addrs.To(core::Dtype::Int64), active_nb_masks,
            block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
            points, normals, colors, block_resolution_, voxel_size_);
    auto pcd = PointCloud(points);
    pcd.SetPointNormals(normals);
    if (colors.NumElements() != 0) {
        pcd.SetPointColors(colors);
    }

    return pcd;
}

TriangleMesh TSDFVoxelGrid::ExtractSurfaceMesh() {
    // Query active blocks and their nearest neighbors to handle boundary cases.
    core::Tensor active_addrs;
    block_hashmap_->GetActiveIndices(active_addrs);
    core::Tensor active_nb_addrs, active_nb_masks;
    std::tie(active_nb_addrs, active_nb_masks) =
            BufferRadiusNeighbors(active_addrs);

    // Map active indices to [0, num_blocks] to be allocated for surface mesh.
    int64_t num_blocks = block_hashmap_->Size();
    core::Tensor inverse_index_map({block_hashmap_->GetCapacity()},
                                   core::Dtype::Int64, device_);
    std::vector<int64_t> iota_map(num_blocks);
    std::iota(iota_map.begin(), iota_map.end(), 0);
    inverse_index_map.IndexSet(
            {active_addrs.To(core::Dtype::Int64)},
            core::Tensor(iota_map, {num_blocks}, core::Dtype::Int64, device_));

    core::Tensor vertices, triangles, vertex_normals, vertex_colors;
    kernel::tsdf::ExtractSurfaceMesh(
            active_addrs.To(core::Dtype::Int64), inverse_index_map,
            active_nb_addrs.To(core::Dtype::Int64), active_nb_masks,
            block_hashmap_->GetKeyTensor(), block_hashmap_->GetValueTensor(),
            vertices, triangles, vertex_normals, vertex_colors,
            block_resolution_, voxel_size_);

    TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexNormals(vertex_normals);
    if (vertex_colors.NumElements() != 0) {
        mesh.SetVertexColors(vertex_colors);
    }
    return mesh;
}

TSDFVoxelGrid TSDFVoxelGrid::To(const core::Device &device, bool copy) const {
    if (!copy && GetDevice() == device) {
        return *this;
    }

    TSDFVoxelGrid device_tsdf_voxelgrid(attr_dtype_map_, voxel_size_,
                                        sdf_trunc_, block_resolution_,
                                        block_count_, device);
    auto device_tsdf_hashmap = device_tsdf_voxelgrid.block_hashmap_;
    *device_tsdf_hashmap = block_hashmap_->To(device);
    return device_tsdf_voxelgrid;
}

std::pair<core::Tensor, core::Tensor> TSDFVoxelGrid::BufferRadiusNeighbors(
        const core::Tensor &active_addrs) {
    // Fixed radius search for spatially hashed voxel blocks.
    // A generalization will be implementing dense/sparse fixed radius search
    // with coordinates as hashmap keys.
    core::Tensor key_buffer_int3_tensor = block_hashmap_->GetKeyTensor();

    core::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
            {active_addrs.To(core::Dtype::Int64)});
    int64_t n = active_keys.GetShape()[0];

    // Fill in radius nearest neighbors.
    core::Tensor keys_nb({27, n, 3}, core::Dtype::Int32, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        core::Tensor dt = core::Tensor(std::vector<int>{dx - 1, dy - 1, dz - 1},
                                       {1, 3}, core::Dtype::Int32, device_);
        keys_nb[nb] = active_keys + dt;
    }
    keys_nb = keys_nb.View({27 * n, 3});

    core::Tensor addrs_nb, masks_nb;
    block_hashmap_->Find(keys_nb, addrs_nb, masks_nb);
    return std::make_pair(addrs_nb.View({27, n, 1}), masks_nb.View({27, n, 1}));
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
