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
#include <vector>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

#if defined(__CUDACC__)
void UnprojectCUDA
#else
void UnprojectCPU
#endif
        (const core::Tensor& depth,
         utility::optional<std::reference_wrapper<const core::Tensor>>
                 image_colors,
         core::Tensor& points,
         utility::optional<std::reference_wrapper<core::Tensor>> colors,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         float depth_scale,
         float depth_max,
         int64_t stride) {

    const bool have_colors = image_colors.has_value();
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer image_colors_indexer;

    core::Tensor pose = t::geometry::InverseTransformation(extrinsics);
    TransformIndexer ti(intrinsics, pose, 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    points = core::Tensor({rows_strided * cols_strided, 3},
                          core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer colors_indexer;
    if (have_colors) {
        const auto& imcol = image_colors.value().get();
        image_colors_indexer = NDArrayIndexer{imcol, 2};
        colors.value().get() =
                core::Tensor({rows_strided * cols_strided, 3},
                             core::Dtype::Float32, imcol.GetDevice());
        colors_indexer = NDArrayIndexer(colors.value().get(), 1);
    }

    // Counter
#if defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       depth.GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    int64_t n = rows_strided * cols_strided;
#if defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif

    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                int64_t workload_idx) {
            int64_t y = (workload_idx / cols_strided) * stride;
            int64_t x = (workload_idx % cols_strided) * stride;

            float d = *depth_indexer.GetDataPtr<scalar_t>(x, y) / depth_scale;
            if (d > 0 && d < depth_max) {
                int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                float x_c = 0, y_c = 0, z_c = 0;
                ti.Unproject(static_cast<float>(x), static_cast<float>(y), d,
                             &x_c, &y_c, &z_c);

                float* vertex = point_indexer.GetDataPtr<float>(idx);
                ti.RigidTransform(x_c, y_c, z_c, vertex + 0, vertex + 1,
                                  vertex + 2);
                if (have_colors) {
                    float* pcd_pixel = colors_indexer.GetDataPtr<float>(idx);
                    float* image_pixel =
                            image_colors_indexer.GetDataPtr<float>(x, y);
                    *pcd_pixel = *image_pixel;
                    *(pcd_pixel + 1) = *(image_pixel + 1);
                    *(pcd_pixel + 2) = *(image_pixel + 2);
                }
            }
        });
    });
#if defined(__CUDACC__)
    int total_pts_count = count.Item<int>();
#else
    int total_pts_count = (*count_ptr).load();
#endif

#ifdef __CUDACC__
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
    points = points.Slice(0, 0, total_pts_count);
    if (have_colors) {
        colors.value().get() =
                colors.value().get().Slice(0, 0, total_pts_count);
    }
}
}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
