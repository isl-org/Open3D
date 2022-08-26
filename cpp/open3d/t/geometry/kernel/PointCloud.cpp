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

#include "open3d/t/geometry/kernel/PointCloud.h"

#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pointcloud {

void Unproject(const core::Tensor& depth,
               utility::optional<std::reference_wrapper<const core::Tensor>>
                       image_colors,
               core::Tensor& points,
               utility::optional<std::reference_wrapper<core::Tensor>> colors,
               const core::Tensor& intrinsics,
               const core::Tensor& extrinsics,
               float depth_scale,
               float depth_max,
               int64_t stride) {
    if (image_colors.has_value() != colors.has_value()) {
        utility::LogError(
                "Both or none of image_colors and colors must have values.");
    }

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor extrinsics_d = extrinsics.To(host, core::Float64).Contiguous();

    const core::Device device = depth.GetDevice();

    if (image_colors.has_value()) {
        core::AssertTensorDevice(image_colors.value(), device);
    }

    if (depth.IsCPU()) {
        UnprojectCPU(depth, image_colors, points, colors, intrinsics_d,
                     extrinsics_d, depth_scale, depth_max, stride);
    } else if (depth.IsCUDA()) {
        CUDA_CALL(UnprojectCUDA, depth, image_colors, points, colors,
                  intrinsics_d, extrinsics_d, depth_scale, depth_max, stride);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void Project(
        core::Tensor& depth,
        utility::optional<std::reference_wrapper<core::Tensor>> image_colors,
        const core::Tensor& points,
        utility::optional<std::reference_wrapper<const core::Tensor>> colors,
        const core::Tensor& intrinsics,
        const core::Tensor& extrinsics,
        float depth_scale,
        float depth_max) {
    if (image_colors.has_value() != colors.has_value()) {
        utility::LogError(
                "Both or none of image_colors and colors must have values.");
    }

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(extrinsics, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor extrinsics_d = extrinsics.To(host, core::Float64).Contiguous();

    const core::Device device = depth.GetDevice();

    if (image_colors.has_value()) {
        core::AssertTensorDevice(image_colors.value(), device);
    }

    if (depth.IsCPU()) {
        ProjectCPU(depth, image_colors, points, colors, intrinsics_d,
                   extrinsics_d, depth_scale, depth_max);
    } else if (depth.IsCUDA()) {
        CUDA_CALL(ProjectCUDA, depth, image_colors, points, colors,
                  intrinsics_d, extrinsics_d, depth_scale, depth_max);
    } else {
        utility::LogError("Unimplemented device");
    }
}

void GetPointMaskWithinAABB(const core::Tensor& points,
                            const core::Tensor& min_bound,
                            const core::Tensor& max_bound,
                            core::Tensor& mask) {
    core::AssertTensorShape(min_bound, {3});
    core::AssertTensorShape(max_bound, {3});
    core::AssertTensorShape(mask, {points.GetLength()});
    // Mask must be a bool tensor.
    core::AssertTensorDtype(mask, core::Bool);

    // Convert points, min_bound and max_bound into contiguous Tensor.
    const core::Tensor min_bound_d = min_bound.Contiguous();
    const core::Tensor max_bound_d = max_bound.Contiguous();
    const core::Tensor points_d = points.Contiguous();

    if (mask.IsCPU()) {
        GetPointMaskWithinAABBCPU(points_d, min_bound_d, max_bound_d, mask);
    } else if (mask.IsCUDA()) {
        CUDA_CALL(GetPointMaskWithinAABBCUDA, points_d, min_bound_d,
                  max_bound_d, mask);
    } else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace pointcloud
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
