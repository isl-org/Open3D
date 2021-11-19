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

#include "open3d/t/pipelines/kernel/RGBDOdometry.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeOdometryResultPointToPlane(
        const core::Tensor &source_vertex_map,
        const core::Tensor &target_vertex_map,
        const core::Tensor &target_normal_map,
        const core::Tensor &intrinsics,
        const core::Tensor &init_source_to_target,
        core::Tensor &delta,
        float &inlier_residual,
        int &inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(target_vertex_map, supported_dtype);
    core::AssertTensorDtype(target_normal_map, supported_dtype);

    core::AssertTensorDevice(target_vertex_map, device);
    core::AssertTensorDevice(target_normal_map, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        ComputeOdometryResultPointToPlaneCPU(
                source_vertex_map, target_vertex_map, target_normal_map,
                intrinsics_d, trans_d, delta, inlier_residual, inlier_count,
                depth_outlier_trunc, depth_huber_delta);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputeOdometryResultPointToPlaneCUDA, source_vertex_map,
                  target_vertex_map, target_normal_map, intrinsics_d, trans_d,
                  delta, inlier_residual, inlier_count, depth_outlier_trunc,
                  depth_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

void ComputeOdometryResultIntensity(const core::Tensor &source_depth,
                                    const core::Tensor &target_depth,
                                    const core::Tensor &source_intensity,
                                    const core::Tensor &target_intensity,
                                    const core::Tensor &target_intensity_dx,
                                    const core::Tensor &target_intensity_dy,
                                    const core::Tensor &source_vertex_map,
                                    const core::Tensor &intrinsics,
                                    const core::Tensor &init_source_to_target,
                                    core::Tensor &delta,
                                    float &inlier_residual,
                                    int &inlier_count,
                                    const float depth_outlier_trunc,
                                    const float intensity_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(source_depth, supported_dtype);
    core::AssertTensorDtype(target_depth, supported_dtype);
    core::AssertTensorDtype(source_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity_dx, supported_dtype);
    core::AssertTensorDtype(target_intensity_dy, supported_dtype);

    core::AssertTensorDevice(source_depth, device);
    core::AssertTensorDevice(target_depth, device);
    core::AssertTensorDevice(source_intensity, device);
    core::AssertTensorDevice(target_intensity, device);
    core::AssertTensorDevice(target_intensity_dx, device);
    core::AssertTensorDevice(target_intensity_dy, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        ComputeOdometryResultIntensityCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_intensity_dx, target_intensity_dy, source_vertex_map,
                intrinsics_d, trans_d, delta, inlier_residual, inlier_count,
                depth_outlier_trunc, intensity_huber_delta);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputeOdometryResultIntensityCUDA, source_depth,
                  target_depth, source_intensity, target_intensity,
                  target_intensity_dx, target_intensity_dy, source_vertex_map,
                  intrinsics_d, trans_d, delta, inlier_residual, inlier_count,
                  depth_outlier_trunc, intensity_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

void ComputeOdometryResultHybrid(const core::Tensor &source_depth,
                                 const core::Tensor &target_depth,
                                 const core::Tensor &source_intensity,
                                 const core::Tensor &target_intensity,
                                 const core::Tensor &target_depth_dx,
                                 const core::Tensor &target_depth_dy,
                                 const core::Tensor &target_intensity_dx,
                                 const core::Tensor &target_intensity_dy,
                                 const core::Tensor &source_vertex_map,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 float &inlier_residual,
                                 int &inlier_count,
                                 const float depth_outlier_trunc,
                                 const float depth_huber_delta,
                                 const float intensity_huber_delta) {
    // Only Float32 is supported as of now. TODO. Support Float64.
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(source_depth, supported_dtype);
    core::AssertTensorDtype(target_depth, supported_dtype);
    core::AssertTensorDtype(source_intensity, supported_dtype);
    core::AssertTensorDtype(target_intensity, supported_dtype);
    core::AssertTensorDtype(target_depth_dx, supported_dtype);
    core::AssertTensorDtype(target_depth_dy, supported_dtype);
    core::AssertTensorDtype(target_intensity_dx, supported_dtype);
    core::AssertTensorDtype(target_intensity_dy, supported_dtype);

    core::AssertTensorDevice(source_depth, device);
    core::AssertTensorDevice(target_depth, device);
    core::AssertTensorDevice(source_intensity, device);
    core::AssertTensorDevice(target_intensity, device);
    core::AssertTensorDevice(target_depth_dx, device);
    core::AssertTensorDevice(target_depth_dy, device);
    core::AssertTensorDevice(target_intensity_dx, device);
    core::AssertTensorDevice(target_intensity_dy, device);

    core::AssertTensorShape(intrinsics, {3, 3});
    core::AssertTensorShape(init_source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsics.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            init_source_to_target.To(host, core::Float64).Contiguous();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        ComputeOdometryResultHybridCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_depth_dx, target_depth_dy, target_intensity_dx,
                target_intensity_dy, source_vertex_map, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta, intensity_huber_delta);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputeOdometryResultHybridCUDA, source_depth, target_depth,
                  source_intensity, target_intensity, target_depth_dx,
                  target_depth_dy, target_intensity_dx, target_intensity_dy,
                  source_vertex_map, intrinsics_d, trans_d, delta,
                  inlier_residual, inlier_count, depth_outlier_trunc,
                  depth_huber_delta, intensity_huber_delta);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
