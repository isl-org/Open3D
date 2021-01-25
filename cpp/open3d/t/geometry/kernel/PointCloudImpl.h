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

#include <atomic>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void UnprojectCUDA
#else
void UnprojectCPU
#endif
        (const core::Tensor& depth,
         core::Tensor& points,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         float depth_scale,
         float depth_max,
         int64_t stride) {

    NDArrayIndexer depth_indexer(depth, 2);
    TransformIndexer ti(intrinsics, extrinsics.Inverse(), 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    points = core::Tensor({rows_strided * cols_strided, 3},
                          core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);

    // Counter
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       depth.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    int64_t n = rows_strided * cols_strided;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                int64_t y = (workload_idx / cols_strided) * stride;
                int64_t x = (workload_idx % cols_strided) * stride;

                float d = *depth_indexer.GetDataPtrFromCoord<uint16_t>(x, y) /
                          depth_scale;
                if (d > 0 && d < depth_max) {
                    int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                    float x_c = 0, y_c = 0, z_c = 0;
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, &x_c, &y_c, &z_c);

                    float* vertex =
                            point_indexer.GetDataPtrFromCoord<float>(idx);
                    ti.RigidTransform(x_c, y_c, z_c, vertex + 0, vertex + 1,
                                      vertex + 2);
                }
            });
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_pts_count = count.Item<int>();
#else
    int total_pts_count = (*count_ptr).load();
#endif
    points = points.Slice(0, 0, total_pts_count);
}
}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
