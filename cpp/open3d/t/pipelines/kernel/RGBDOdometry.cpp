// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

    if (device.IsCPU()) {
        ComputeOdometryResultPointToPlaneCPU(
                source_vertex_map, target_vertex_map, target_normal_map,
                intrinsics_d, trans_d, delta, inlier_residual, inlier_count,
                depth_outlier_trunc, depth_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_vertex_map.GetDevice());
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

    if (device.IsCPU()) {
        ComputeOdometryResultIntensityCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_intensity_dx, target_intensity_dy, source_vertex_map,
                intrinsics_d, trans_d, delta, inlier_residual, inlier_count,
                depth_outlier_trunc, intensity_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_depth.GetDevice());
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

    if (device.IsCPU()) {
        ComputeOdometryResultHybridCPU(
                source_depth, target_depth, source_intensity, target_intensity,
                target_depth_dx, target_depth_dy, target_intensity_dx,
                target_intensity_dy, source_vertex_map, intrinsics_d, trans_d,
                delta, inlier_residual, inlier_count, depth_outlier_trunc,
                depth_huber_delta, intensity_huber_delta);
    } else if (device.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_depth.GetDevice());
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

void ComputeOdometryInformationMatrix(const core::Tensor &source_vertex_map,
                                      const core::Tensor &target_vertex_map,
                                      const core::Tensor &intrinsic,
                                      const core::Tensor &source_to_target,
                                      const float square_dist_thr,
                                      core::Tensor &information) {
    core::AssertTensorDtypes(source_vertex_map, {core::Float32});

    const core::Dtype supported_dtype = source_vertex_map.GetDtype();
    const core::Device device = source_vertex_map.GetDevice();

    core::AssertTensorDtype(target_vertex_map, supported_dtype);
    core::AssertTensorDevice(target_vertex_map, device);

    core::AssertTensorShape(intrinsic, {3, 3});
    core::AssertTensorShape(source_to_target, {4, 4});

    static const core::Device host("CPU:0");
    core::Tensor intrinsics_d = intrinsic.To(host, core::Float64).Contiguous();
    core::Tensor trans_d =
            source_to_target.To(host, core::Float64).Contiguous();

    if (device.GetType() == core::Device::DeviceType::CPU) {
        ComputeOdometryInformationMatrixCPU(
                source_vertex_map, target_vertex_map, intrinsic,
                source_to_target, square_dist_thr, information);
    } else if (device.GetType() == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputeOdometryInformationMatrixCUDA, source_vertex_map,
                  target_vertex_map, intrinsic, source_to_target,
                  square_dist_thr, information);
    } else {
        utility::LogError("Unimplemented device.");
    }
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
