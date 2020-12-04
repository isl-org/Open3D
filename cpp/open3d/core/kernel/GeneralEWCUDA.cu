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
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/core/kernel/GeneralEW.h"
#include "open3d/core/kernel/GeneralIndexer.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

void CUDAUnprojectKernel(const std::unordered_map<std::string, Tensor>& srcs,
                         std::unordered_map<std::string, Tensor>& dsts) {
    if (srcs.count("depth") == 0 || srcs.count("intrinsics") == 0 ||
        srcs.count("depth_scale") == 0) {
        utility::LogError(
                "[GeneralEWCUDA] expect depth, intrinsics, and depth_scale as "
                "input");
    }

    // Input
    Tensor depth = srcs.at("depth").To(core::Dtype::Float32);
    Tensor intrinsics = srcs.at("intrinsics").To(core::Dtype::Float32);
    float depth_scale = srcs.at("depth_scale").Item<float>();

    NDArrayIndexer depth_ndi(depth, 2);
    TransformIndexer ti(intrinsics);

    // Output
    Tensor vertex_map({depth_ndi.GetShape(0), depth_ndi.GetShape(1), 3},
                      core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer vertex_ndi(vertex_map, 2);

    // Workload
    int64_t n = depth_ndi.NumElements();

    CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
                int64_t y, x;
                depth_ndi.WorkloadToCoord(workload_idx, &x, &y);

                float d = *static_cast<float*>(depth_ndi.GetDataPtrFromWorkload(
                                  workload_idx)) /
                          depth_scale;

                float* vertex = static_cast<float*>(
                        vertex_ndi.GetDataPtrFromWorkload(workload_idx));

                ti.Unproject(static_cast<float>(x), static_cast<float>(y), d,
                             vertex, vertex + 1, vertex + 2);
            });

    dsts.emplace("vertex_map", vertex_map);
}

void GeneralEWCUDA(const std::unordered_map<std::string, Tensor>& srcs,
                   std::unordered_map<std::string, Tensor>& dsts,
                   GeneralEWOpCode op_code) {
    switch (op_code) {
        case GeneralEWOpCode::Unproject:
            CUDAUnprojectKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFIntegrate:
            break;
        case GeneralEWOpCode::TSDFSurfaceExtraction:
            break;
        case GeneralEWOpCode::MarchingCubesPass0:
            break;
        case GeneralEWOpCode::MarchingCubesPass1:
            break;
        case GeneralEWOpCode::MarchingCubesPass2:
            break;
        case GeneralEWOpCode::RayCasting:
            break;
        case GeneralEWOpCode::Debug: {
            int64_t n = 10;
            CUDALauncher::LaunchGeneralKernel(
                    n, [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {});
            break;
        }
        default:
            break;
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
