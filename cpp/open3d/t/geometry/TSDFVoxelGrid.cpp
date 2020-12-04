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

#include "open3d/core/kernel/Kernel.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

TSDFVoxelGrid::TSDFVoxelGrid(
        std::unordered_map<std::string, int> attr_channel_map,
        float voxel_size,
        float sdf_trunc,
        int64_t block_resolution,
        int64_t block_count,
        const core::Device &device)
    : attr_channel_map_(attr_channel_map),
      voxel_size_(voxel_size),
      sdf_trunc_(sdf_trunc),
      block_resolution_(block_resolution),
      block_count_(block_count),
      device_(device) {
    int total_channels = 0;
    for (auto &kv : attr_channel_map_) {
        total_channels += kv.second;
    }
    core::Dtype key_dtype(core::Dtype::DtypeCode::Object,
                          core::Dtype::Int64.ByteSize() * 3, "_hash_k");
    core::Dtype val_dtype(core::Dtype::DtypeCode::Object,
                          core::Dtype::Float32.ByteSize() * block_resolution_ *
                                  block_resolution_ * block_resolution_ *
                                  total_channels,
                          "_hash_v");
    block_hashmap_ = std::make_shared<core::Hashmap>(block_count_, key_dtype,
                                                     val_dtype, device);
}

void TSDFVoxelGrid::Integrate(const Image &depth,
                              const core::Tensor &intrinsics,
                              const core::Tensor &extrinsics,
                              double depth_scale) {
    // Unproject
    PointCloud pcd =
            PointCloud::CreateFromDepthImage(depth, intrinsics, depth_scale);
    pcd.Transform(extrinsics.Inverse());

    float block_size = voxel_size_ * block_resolution_;
    PointCloud pcd_down = pcd.VoxelDownSample(voxel_size_ * block_resolution_);

    // TODO: reuse code in VoxelDownSample
    core::Tensor block_coords = (pcd_down.GetPoints().AsTensor() / block_size)
                                        .To(core::Dtype::Int64);
    core::Tensor addrs, masks;
    block_hashmap_->Activate(block_coords, addrs, masks);
    block_hashmap_->Find(block_coords, addrs, masks);

    // Input
    std::unordered_map<std::string, core::Tensor> srcs = {
            {"depth", depth.AsTensor()},
            {"indices", addrs.To(core::Dtype::Int64).IndexGet({masks})},
            {"block_keys",
             core::Hashmap::ReinterpretBufferTensor(
                     block_hashmap_->GetKeyTensor(),
                     {block_hashmap_->GetCapacity(), 3}, core::Dtype::Int64)},
            {"intrinsics", intrinsics.Copy(device_)},
            {"extrinsics", extrinsics.Copy(device_)},
            {"resolution", core::Tensor(std::vector<int64_t>{block_resolution_},
                                        {}, core::Dtype::Int64, device_)},
            {"depth_scale",
             core::Tensor(std::vector<float>{static_cast<float>(depth_scale)},
                          {}, core::Dtype::Float32, device_)},
            {"voxel_size", core::Tensor(std::vector<float>{voxel_size_}, {},
                                        core::Dtype::Float32, device_)},
            {"sdf_trunc", core::Tensor(std::vector<float>{sdf_trunc_}, {},
                                       core::Dtype::Float32, device_)}};

    // In-place modified output
    std::unordered_map<std::string, core::Tensor> dsts = {
            {"block_values",
             core::Hashmap::ReinterpretBufferTensor(
                     block_hashmap_->GetValueTensor(),
                     {block_hashmap_->GetCapacity(), block_resolution_,
                      block_resolution_, block_resolution_, 2},
                     core::Dtype::Float32)}};

    core::kernel::GeneralEW(srcs, dsts,
                            core::kernel::GeneralEWOpCode::TSDFIntegrate);
    utility::LogInfo("Active blocks = {}", block_hashmap_->Size());
}

std::pair<core::Tensor, core::Tensor> TSDFVoxelGrid::BufferRadiusNeighbors() {
    core::Tensor active_addrs;
    block_hashmap_->GetActiveIndices(
            static_cast<core::addr_t *>(active_addrs.GetDataPtr()));
    core::Tensor key_buffer_int3_tensor =
            block_hashmap_->ReinterpretBufferTensor(
                    block_hashmap_->GetKeyTensor(),
                    {block_hashmap_->GetCapacity(), 3}, core::Dtype::Int64);

    core::Tensor active_keys = key_buffer_int3_tensor.IndexGet(
            {active_addrs.To(core::Dtype::Int64)});
    int64_t n = active_keys.GetShape()[0];

    // Fill in radius nearest neighbors
    core::Tensor keys_nb({27, n, 3}, core::Dtype::Int64, device_);
    for (int nb = 0; nb < 27; ++nb) {
        int dz = nb / 9;
        int dy = (nb % 9) / 3;
        int dx = nb % 3;
        core::Tensor dt =
                core::Tensor(std::vector<int64_t>{dx - 1, dy - 1, dz - 1},
                             {1, 3}, core::Dtype::Int64, device_);
        keys_nb[nb] = active_keys + dt;
    }

    core::Tensor addrs_nb, masks_nb;
    block_hashmap_->Find(keys_nb, addrs_nb, masks_nb);
    return std::make_pair(addrs_nb, masks_nb);
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
