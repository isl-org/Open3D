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
#include "open3d/core/kernel/Kernel.h"
#include "open3d/t/geometry/PointCloud.h"
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
                              double depth_scale,
                              double depth_max) {
    Image empty_color;
    Integrate(depth, empty_color, intrinsics, extrinsics, depth_scale,
              depth_max);
}

void TSDFVoxelGrid::Integrate(const Image &depth,
                              const Image &color,
                              const core::Tensor &intrinsics,
                              const core::Tensor &extrinsics,
                              double depth_scale,
                              double depth_max) {
    if (depth.IsEmpty()) {
        utility::LogError(
                "[TSDFVoxelGrid] input depth is empty for integration.");
    }

    // Create a point cloud from a low-resolution depth input to roughly
    // estimate surfaces.
    PointCloud pcd = PointCloud::CreateFromDepthImage(
            depth, intrinsics, extrinsics, depth_scale, depth_max, 4);

    // Determine voxel blocks to allocate.
    std::unordered_map<std::string, core::Tensor> srcs = {
            {"points", pcd.GetPoints().Contiguous()},
            {"resolution", core::Tensor(std::vector<int64_t>{block_resolution_},
                                        {}, core::Dtype::Int64, device_)},
            {"voxel_size", core::Tensor(std::vector<float>{voxel_size_}, {},
                                        core::Dtype::Float32, device_)},
            {"sdf_trunc", core::Tensor(std::vector<float>{sdf_trunc_}, {},
                                       core::Dtype::Float32, device_)}};
    std::unordered_map<std::string, core::Tensor> dsts;

    core::kernel::GeneralEW(srcs, dsts,
                            core::kernel::GeneralEWOpCode::TSDFTouch);
    if (dsts.count("block_coords") == 0) {
        utility::LogError(
                "[TSDFVoxelGrid] touch launch failed, expected block_coords");
    }

    // Active voxel blocks in the block hashmap.
    core::Tensor block_coords = dsts.at("block_coords");
    core::Tensor addrs, masks;
    int64_t n = block_hashmap_->Size();
    try {
        block_hashmap_->Activate(block_coords, addrs, masks);
    } catch (const std::runtime_error &error) {
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

    // TSDF Integration.
    srcs = {{"depth", depth.AsTensor().Contiguous()},
            {"indices", addrs.To(core::Dtype::Int64).IndexGet({masks})},
            {"block_keys", block_hashmap_->GetKeyTensor()},
            {"intrinsics", intrinsics.Copy(device_)},
            {"extrinsics", extrinsics.Copy(device_)},
            {"resolution", core::Tensor(std::vector<int64_t>{block_resolution_},
                                        {}, core::Dtype::Int64, device_)},
            {"depth_scale",
             core::Tensor(std::vector<float>{static_cast<float>(depth_scale)},
                          {}, core::Dtype::Float32, device_)},
            {"depth_max",
             core::Tensor(std::vector<float>{static_cast<float>(depth_max)}, {},
                          core::Dtype::Float32, device_)},
            {"voxel_size", core::Tensor(std::vector<float>{voxel_size_}, {},
                                        core::Dtype::Float32, device_)},
            {"sdf_trunc", core::Tensor(std::vector<float>{sdf_trunc_}, {},
                                       core::Dtype::Float32, device_)}};

    if (color.IsEmpty()) {
        utility::LogDebug(
                "[TSDFIntegrate] color image is empty, perform depth "
                "integration only.");
    } else if (color.GetRows() == depth.GetRows() &&
               color.GetCols() == depth.GetCols() && color.GetChannels() == 3) {
        if (attr_dtype_map_.count("color") != 0) {
            srcs.emplace(
                    "color",
                    color.AsTensor().To(core::Dtype::Float32).Contiguous());
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

    dsts = {{"block_values", block_hashmap_->GetValueTensor()}};
    core::kernel::GeneralEW(srcs, dsts,
                            core::kernel::GeneralEWOpCode::TSDFIntegrate);
}

PointCloud TSDFVoxelGrid::ExtractSurfacePoints() {
    // Extract active voxel blocks from the hashmap.
    core::Tensor active_addrs;
    block_hashmap_->GetActiveIndices(active_addrs);
    core::Tensor active_nb_addrs, active_nb_masks;
    std::tie(active_nb_addrs, active_nb_masks) =
            BufferRadiusNeighbors(active_addrs);

    // Extract points around zero-crossings.
    std::unordered_map<std::string, core::Tensor> srcs = {
            {"indices", active_addrs.To(core::Dtype::Int64)},
            {"nb_indices", active_nb_addrs.To(core::Dtype::Int64)},
            {"nb_masks", active_nb_masks},
            {"block_keys", block_hashmap_->GetKeyTensor()},
            {"block_values", block_hashmap_->GetValueTensor()},
            {"resolution", core::Tensor(std::vector<int64_t>{block_resolution_},
                                        {}, core::Dtype::Int64, device_)},
            {"voxel_size", core::Tensor(std::vector<float>{voxel_size_}, {},
                                        core::Dtype::Float32, device_)}};

    std::unordered_map<std::string, core::Tensor> dsts;
    core::kernel::GeneralEW(srcs, dsts,
                            core::kernel::GeneralEWOpCode::TSDFPointExtraction);
    if (dsts.count("points") == 0) {
        utility::LogError(
                "[TSDFVoxelGrid] extract surface launch failed, points "
                "expected to return.");
    }
    auto pcd = PointCloud(dsts.at("points"));
    pcd.SetPointNormals(dsts.at("normals"));
    if (attr_dtype_map_.count("color") != 0) {
        pcd.SetPointColors(dsts.at("colors"));
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

    std::unordered_map<std::string, core::Tensor> srcs = {
            {"indices", active_addrs.To(core::Dtype::Int64)},
            {"inv_indices", inverse_index_map},
            {"nb_indices", active_nb_addrs.To(core::Dtype::Int64)},
            {"nb_masks", active_nb_masks},
            {"block_keys", block_hashmap_->GetKeyTensor()},
            {"block_values", block_hashmap_->GetValueTensor()},
            {"resolution", core::Tensor(std::vector<int64_t>{block_resolution_},
                                        {}, core::Dtype::Int64, device_)},
            {"voxel_size", core::Tensor(std::vector<float>{voxel_size_}, {},
                                        core::Dtype::Float32, device_)}};

    std::unordered_map<std::string, core::Tensor> dsts;

    core::kernel::GeneralEW(srcs, dsts,
                            core::kernel::GeneralEWOpCode::TSDFMeshExtraction);

    TriangleMesh mesh(dsts.at("vertices"), dsts.at("triangles"));
    mesh.SetVertexNormals(dsts.at("normals"));
    if (attr_dtype_map_.count("color") != 0) {
        mesh.SetVertexColors(dsts.at("colors"));
    }
    return mesh;
}

TSDFVoxelGrid TSDFVoxelGrid::Copy(const core::Device &device) {
    TSDFVoxelGrid device_tsdf_voxelgrid(attr_dtype_map_, voxel_size_,
                                        sdf_trunc_, block_resolution_,
                                        block_count_, device);
    auto device_tsdf_hashmap = device_tsdf_voxelgrid.block_hashmap_;
    *device_tsdf_hashmap = block_hashmap_->Copy(device);
    return device_tsdf_voxelgrid;
}

TSDFVoxelGrid TSDFVoxelGrid::CPU() {
    if (GetDevice().GetType() == core::Device::DeviceType::CPU) {
        return *this;
    }
    return Copy(core::Device("CPU:0"));
}

TSDFVoxelGrid TSDFVoxelGrid::CUDA(int device_id) {
    core::Device device =
            core::Device(core::Device::DeviceType::CUDA, device_id);
    if (GetDevice() == device) {
        return *this;
    }
    return Copy(device);
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
