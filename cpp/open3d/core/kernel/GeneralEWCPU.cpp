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

#include <tbb/concurrent_unordered_set.h>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/kernel/GeneralEW.h"
#include "open3d/core/kernel/GeneralEWMacros.h"
#include "open3d/core/kernel/GeneralEWSharedImpl.h"
#include "open3d/core/kernel/GeneralIndexer.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

void CPUArangeKernel(const std::unordered_map<std::string, Tensor>& srcs,
                     std::unordered_map<std::string, Tensor>& dsts) {
    static std::vector<std::string> src_attrs = {"start", "step"};
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError("Expected Tensor {} in srcs, but did not receive",
                              k);
        }
    }
    if (dsts.count("arange") == 0) {
        utility::LogError(
                "Expected Tensor arange in dsts, but did not receive");
    }

    Tensor start = srcs.at("start");
    Tensor step = srcs.at("step");
    Tensor dst = dsts.at("arange");

    Dtype dtype = srcs.at("start").GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t sstart = start.Item<scalar_t>();
        scalar_t sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
        int64_t n = dst.GetLength();
        CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
            dst_ptr[workload_idx] =
                    sstart + static_cast<scalar_t>(sstep * workload_idx);
        });
    });
}

struct Coord3i {
    Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

struct Coord3iHash {
    size_t operator()(const Coord3i& k) const {
        static const size_t p0 = 73856093;
        static const size_t p1 = 19349669;
        static const size_t p2 = 83492791;

        return (static_cast<size_t>(k.x_) * p0) ^
               (static_cast<size_t>(k.y_) * p1) ^
               (static_cast<size_t>(k.z_) * p2);
    }
};

void CPUTSDFTouchKernel(const std::unordered_map<std::string, Tensor>& srcs,
                        std::unordered_map<std::string, Tensor>& dsts) {
    static std::vector<std::string> src_attrs = {"points", "voxel_size",
                                                 "resolution", "sdf_trunc"};

    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDATSDFTouchKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor pcd = srcs.at("points");
    float voxel_size = srcs.at("voxel_size").Item<float>();
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    float block_size = voxel_size * resolution;

    float sdf_trunc = srcs.at("sdf_trunc").Item<float>();

    int64_t n = pcd.GetLength();
    float* pcd_ptr = static_cast<float*>(pcd.GetDataPtr());

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        float x = pcd_ptr[3 * workload_idx + 0];
        float y = pcd_ptr[3 * workload_idx + 1];
        float z = pcd_ptr[3 * workload_idx + 2];

        int xb_lo = static_cast<int>(std::floor((x - sdf_trunc) / block_size));
        int xb_hi = static_cast<int>(std::floor((x + sdf_trunc) / block_size));
        int yb_lo = static_cast<int>(std::floor((y - sdf_trunc) / block_size));
        int yb_hi = static_cast<int>(std::floor((y + sdf_trunc) / block_size));
        int zb_lo = static_cast<int>(std::floor((z - sdf_trunc) / block_size));
        int zb_hi = static_cast<int>(std::floor((z + sdf_trunc) / block_size));
        for (int xb = xb_lo; xb <= xb_hi; ++xb) {
            for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                    set.emplace(xb, yb, zb);
                }
            }
        }
    });

    int64_t block_count = set.size();
    if (block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, abort integration. Please "
                "check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    core::Tensor block_coords({block_count, 3}, core::Dtype::Int32,
                              pcd.GetDevice());
    int* block_coords_ptr = static_cast<int*>(block_coords.GetDataPtr());
    int count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        int64_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<int>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<int>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<int>(it->z_);
    }

    dsts.emplace("block_coords", block_coords);
}

void GeneralEWCPU(const std::unordered_map<std::string, Tensor>& srcs,
                  std::unordered_map<std::string, Tensor>& dsts,
                  GeneralEWOpCode op_code) {
    switch (op_code) {
        case GeneralEWOpCode::Arange:
            CPUArangeKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::Unproject:
            CPUUnprojectKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFTouch:
            CPUTSDFTouchKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFIntegrate:
            CPUTSDFIntegrateKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFPointExtraction:
            CPUPointExtractionKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFMeshExtraction:
            CPUMeshExtractionKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::RayCasting:
            utility::LogError("[RayCasting] Unimplemented.");
            break;
        default:
            utility::LogError("Unimplemented.");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
