// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/TransformationConverter.h"

#include <cmath>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/t/pipelines/kernel/TransformationConverterImpl.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor RtToTransformation(const core::Tensor &R, const core::Tensor &t) {
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(t, {3});
    core::AssertTensorDtypes(R, {core::Float32, core::Float64});

    const core::Device device = R.GetDevice();
    const core::Dtype dtype = R.GetDtype();

    core::AssertTensorDtype(t, dtype);
    core::AssertTensorDevice(t, device);

    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);

    // Rotation.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(0, 3, 1)},
            R);
    // Translation.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            t.Reshape({3, 1}));
    // Scale [assumed to be 1].
    transformation[3][3] = 1;
    return transformation;
}

template <typename scalar_t>
static void PoseToTransformationDevice(
        core::Tensor &transformation,
        const core::Tensor &pose,
        const core::Device::DeviceType &device_type) {
    scalar_t *transformation_ptr = transformation.GetDataPtr<scalar_t>();
    const scalar_t *pose_ptr = pose.GetDataPtr<scalar_t>();

    if (device_type == core::Device::DeviceType::CPU) {
        PoseToTransformationImpl<scalar_t>(transformation_ptr, pose_ptr);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        core::CUDAScopedDevice scoped_device(transformation.GetDevice());
        PoseToTransformationCUDA<scalar_t>(transformation_ptr, pose_ptr);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
}

core::Tensor PoseToTransformation(const core::Tensor &pose) {
    core::AssertTensorShape(pose, {6});
    core::AssertTensorDtypes(pose, {core::Float32, core::Float64});

    const core::Device device = pose.GetDevice();
    const core::Dtype dtype = pose.GetDtype();
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    transformation = transformation.Contiguous();
    core::Tensor pose_ = pose.Contiguous();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        core::Device::DeviceType device_type = device.GetType();
        PoseToTransformationDevice<scalar_t>(transformation, pose_,
                                             device_type);
    });

    // Translation from pose.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            pose_.GetItem({core::TensorKey::Slice(3, 6, 1)}).Reshape({3, 1}));
    // Scale [assumed to be 1].
    transformation[3][3] = 1;
    return transformation;
}

template <typename scalar_t>
static void TransformationToPoseDevice(
        core::Tensor &pose,
        const core::Tensor &transformation,
        const core::Device::DeviceType &device_type) {
    scalar_t *pose_ptr = pose.GetDataPtr<scalar_t>();
    const scalar_t *transformation_ptr = transformation.GetDataPtr<scalar_t>();

    if (device_type == core::Device::DeviceType::CPU) {
        TransformationToPoseImpl<scalar_t>(pose_ptr, transformation_ptr);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        TransformationToPoseCUDA<scalar_t>(pose_ptr, transformation_ptr);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
}

core::Tensor TransformationToPose(const core::Tensor &transformation) {
    core::AssertTensorShape(transformation, {4, 4});
    core::AssertTensorDtypes(transformation, {core::Float32, core::Float64});

    const core::Device device = transformation.GetDevice();
    const core::Dtype dtype = transformation.GetDtype();
    core::Tensor pose = core::Tensor::Zeros({6}, dtype, device);
    pose = pose.Contiguous();
    core::Tensor transformation_ = transformation.Contiguous();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        core::Device::DeviceType device_type = device.GetType();
        TransformationToPoseDevice<scalar_t>(pose, transformation_,
                                             device_type);
    });

    // Set translation parameters in pose vector.
    pose.SetItem(core::TensorKey::Slice(3, 6, 1),
                 transformation_
                         .GetItem({core::TensorKey::Slice(0, 3, 1),
                                   core::TensorKey::Slice(3, 4, 1)})
                         .Flatten());

    return pose;
}

void DecodeAndSolve6x6(const core::Tensor &A_reduction,
                       core::Tensor &delta,
                       float &inlier_residual,
                       int &inlier_count) {
    const core::Device host(core::Device("CPU:0"));
    core::Tensor A_1x29_host = A_reduction.To(host, core::Float64);
    core::AssertTensorShape(A_reduction, {29});

    double *A_1x29_ptr = A_1x29_host.GetDataPtr<double>();

    core::Tensor AtA = core::Tensor::Empty({6, 6}, core::Float64, host);
    core::Tensor Atb = core::Tensor::Empty({6}, core::Float64, host);

    double *AtA_local_ptr = AtA.GetDataPtr<double>();
    double *Atb_local_ptr = Atb.GetDataPtr<double>();

    for (int j = 0; j < 6; j++) {
        Atb_local_ptr[j] = A_1x29_ptr[21 + j];
        const int64_t reduction_idx = ((j * (j + 1)) / 2);
        for (int k = 0; k <= j; k++) {
            AtA_local_ptr[j * 6 + k] = A_1x29_ptr[reduction_idx + k];
            AtA_local_ptr[k * 6 + j] = A_1x29_ptr[reduction_idx + k];
        }
    }

    // Solve on CPU with double to ensure precision.
    try {
        delta = AtA.Solve(Atb.Neg());
        inlier_residual = A_1x29_ptr[27];
        inlier_count = static_cast<int>(A_1x29_ptr[28]);
    } catch (const std::runtime_error &) {
        utility::LogError(
                "Singular 6x6 linear system detected, tracking failed.");
        delta.Fill(0);
        inlier_residual = 0;
        inlier_count = 0;
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
