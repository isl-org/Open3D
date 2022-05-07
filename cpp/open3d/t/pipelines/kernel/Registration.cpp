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

#include "open3d/t/pipelines/kernel/Registration.h"

#include "open3d/core/TensorCheck.h"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor ComputePosePointToPlane(const core::Tensor &source_points,
                                     const core::Tensor &target_points,
                                     const core::Tensor &target_normals,
                                     const core::Tensor &correspondence_indices,
                                     const registration::RobustKernel &kernel) {
    const core::Device device = source_points.GetDevice();

    // Pose {6,} tensor [output].
    core::Tensor pose = core::Tensor::Empty({6}, core::Float64, device);

    float residual = 0;
    int inlier_count = 0;

    const core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputePosePointToPlaneCPU(
                source_points.Contiguous(), target_points.Contiguous(),
                target_normals.Contiguous(),
                correspondence_indices.Contiguous(), pose, residual,
                inlier_count, source_points.GetDtype(), device, kernel);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputePosePointToPlaneCUDA, source_points.Contiguous(),
                  target_points.Contiguous(), target_normals.Contiguous(),
                  correspondence_indices.Contiguous(), pose, residual,
                  inlier_count, source_points.GetDtype(), device, kernel);
    } else {
        utility::LogError("Unimplemented device.");
    }

    utility::LogDebug("PointToPlane Transform: residual {}, inlier_count {}",
                      residual, inlier_count);

    return pose;
}

core::Tensor ComputePoseColoredICP(const core::Tensor &source_points,
                                   const core::Tensor &source_colors,
                                   const core::Tensor &target_points,
                                   const core::Tensor &target_normals,
                                   const core::Tensor &target_colors,
                                   const core::Tensor &target_color_gradients,
                                   const core::Tensor &correspondence_indices,
                                   const registration::RobustKernel &kernel,
                                   const double &lambda_geometric) {
    const core::Device device = source_points.GetDevice();

    // Pose {6,} tensor [output].
    core::Tensor pose = core::Tensor::Empty({6}, core::Dtype::Float64, device);

    float residual = 0;
    int inlier_count = 0;

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputePoseColoredICPCPU(
                source_points.Contiguous(), source_colors.Contiguous(),
                target_points.Contiguous(), target_normals.Contiguous(),
                target_colors.Contiguous(), target_color_gradients.Contiguous(),
                correspondence_indices.Contiguous(), pose, residual,
                inlier_count, source_points.GetDtype(), device, kernel,
                lambda_geometric);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputePoseColoredICPCUDA, source_points.Contiguous(),
                  source_colors.Contiguous(), target_points.Contiguous(),
                  target_normals.Contiguous(), target_colors.Contiguous(),
                  target_color_gradients.Contiguous(),
                  correspondence_indices.Contiguous(), pose, residual,
                  inlier_count, source_points.GetDtype(), device, kernel,
                  lambda_geometric);
    } else {
        utility::LogError("Unimplemented device.");
    }

    utility::LogDebug("PointToPlane Transform: residual {}, inlier_count {}",
                      residual, inlier_count);

    return pose;
}

std::tuple<core::Tensor, core::Tensor> ComputeRtPointToPoint(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &correspondence_indices) {
    const core::Device device = source_points.GetDevice();

    // [Output] Rotation and translation tensor of type Float64.
    core::Tensor R, t;

    int inlier_count = 0;

    const core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        // Pointer to point cloud data - indexed according to correspondences.
        ComputeRtPointToPointCPU(
                source_points.Contiguous(), target_points.Contiguous(),
                correspondence_indices.Contiguous(), R, t, inlier_count,
                source_points.GetDtype(), device);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        // TODO: Implement optimized CUDA reduction kernel.
        core::Tensor valid = correspondence_indices.Ne(-1).Reshape({-1});
        // correpondence_set : (i, corres[i]).

        if (valid.GetLength() == 0) {
            utility::LogError("No valid correspondence present.");
        }

        // source[i] and target[corres[i]] is a correspondence.
        core::Tensor source_indices =
                core::Tensor::Arange(0, source_points.GetShape()[0], 1,
                                     core::Int64, device)
                        .IndexGet({valid});
        // Only take valid indices.
        core::Tensor target_indices =
                correspondence_indices.IndexGet({valid}).Reshape({-1});

        // Number of good correspondences (C).
        inlier_count = source_indices.GetLength();

        core::Tensor source_select = source_points.IndexGet({source_indices});
        core::Tensor target_select = target_points.IndexGet({target_indices});

        // https://ieeexplore.ieee.org/document/88573
        core::Tensor mean_s = source_select.Mean({0}, true);
        core::Tensor mean_t = target_select.Mean({0}, true);

        // Compute linear system on CPU as Float64.
        core::Device host("CPU:0");
        core::Tensor Sxy = (target_select - mean_t)
                                   .T()
                                   .Matmul(source_select - mean_s)
                                   .Div_(static_cast<float>(inlier_count))
                                   .To(host, core::Float64);

        mean_s = mean_s.To(host, core::Float64);
        mean_t = mean_t.To(host, core::Float64);

        core::Tensor U, D, VT;
        std::tie(U, D, VT) = Sxy.SVD();
        core::Tensor S = core::Tensor::Eye(3, core::Float64, host);
        if (U.Det() * (VT.T()).Det() < 0) {
            S[-1][-1] = -1;
        }
        R = U.Matmul(S.Matmul(VT));
        t = mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1});
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
    return std::make_tuple(R, t);
}

core::Tensor ComputeInformationMatrix(
        const core::Tensor &target_points,
        const core::Tensor &correspondence_indices) {
    const core::Device device = target_points.GetDevice();

    core::Tensor information_matrix =
            core::Tensor::Empty({6, 6}, core::Float64, core::Device("CPU:0"));

    const core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputeInformationMatrixCPU(
                target_points.Contiguous(), correspondence_indices.Contiguous(),
                information_matrix, target_points.GetDtype(), device);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputeInformationMatrixCUDA, target_points.Contiguous(),
                  correspondence_indices.Contiguous(), information_matrix,
                  target_points.GetDtype(), device);
    } else {
        utility::LogError("Unimplemented device.");
    }

    return information_matrix;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
