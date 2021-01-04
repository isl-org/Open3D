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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/MarchingCubesMacros.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGridShared.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {

struct Coord3i {
    OPEN3D_HOST_DEVICE Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool OPEN3D_HOST_DEVICE operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

void CUDATSDFTouchKernel(
        const std::unordered_map<std::string, core::Tensor>& srcs,
        std::unordered_map<std::string, core::Tensor>& dsts) {
    static std::vector<std::string> src_attrs = {
            "points",
            "voxel_size",
            "resolution",
    };

    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDATSDFTouchKernel] expected core::Tensor {} in srcs, "
                    "but "
                    "did not receive",
                    k);
        }
    }

    core::Tensor pcd = srcs.at("points");
    float voxel_size = srcs.at("voxel_size").Item<float>();
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    float block_size = voxel_size * resolution;

    float sdf_trunc = srcs.at("sdf_trunc").Item<float>();

    core::Device device = pcd.GetDevice();

    int64_t n = pcd.GetLength();
    float* pcd_ptr = static_cast<float*>(pcd.GetDataPtr());

    core::Tensor block_coordi({8 * n, 3}, core::Dtype::Int32, device);
    int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32, device);
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float x = pcd_ptr[3 * workload_idx + 0];
                float y = pcd_ptr[3 * workload_idx + 1];
                float z = pcd_ptr[3 * workload_idx + 2];

                int xb_lo =
                        static_cast<int>(floor((x - sdf_trunc) / block_size));
                int xb_hi =
                        static_cast<int>(floor((x + sdf_trunc) / block_size));
                int yb_lo =
                        static_cast<int>(floor((y - sdf_trunc) / block_size));
                int yb_hi =
                        static_cast<int>(floor((y + sdf_trunc) / block_size));
                int zb_lo =
                        static_cast<int>(floor((z - sdf_trunc) / block_size));
                int zb_hi =
                        static_cast<int>(floor((z + sdf_trunc) / block_size));

                for (int xb = xb_lo; xb <= xb_hi; ++xb) {
                    for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                        for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                            int idx = atomicAdd(count_ptr, 1);
                            block_coordi_ptr[3 * idx + 0] = xb;
                            block_coordi_ptr[3 * idx + 1] = yb;
                            block_coordi_ptr[3 * idx + 2] = zb;
                        }
                    }
                }
            });

    int total_block_count = count.Item<int>();
    if (total_block_count == 0) {
        utility::LogError(
                "[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Hashmap pcd_block_hashmap(total_block_count, core::Dtype::Int32,
                                    core::Dtype::Int32, {3}, {1}, device);
    core::Tensor block_addrs, block_masks;
    pcd_block_hashmap.Activate(block_coordi.Slice(0, 0, count.Item<int>()),
                               block_addrs, block_masks);
    dsts.emplace("block_coords", block_coordi.IndexGet({block_masks}));
}

void GeneralEWCUDA(const std::unordered_map<std::string, core::Tensor>& srcs,
                   std::unordered_map<std::string, core::Tensor>& dsts,
                   GeneralEWOpCode op_code) {
    switch (op_code) {
        case GeneralEWOpCode::Unproject:
            CUDAUnprojectKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFTouch:
            CUDATSDFTouchKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFIntegrate:
            CUDATSDFIntegrateKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFPointExtraction:
            CUDAPointExtractionKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFMeshExtraction:
            CUDAMeshExtractionKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::RayCasting:
            utility::LogError("[RayCasting] Unimplemented.");
            break;
        default:
            break;
    }
}

}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
