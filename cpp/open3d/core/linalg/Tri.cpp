// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/Tri.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/TriImpl.h"

namespace open3d {
namespace core {

static void CheckInput(const Tensor& A, const int diagonal) {
    // Check dimensions.
    SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D.", A_shape.size());
    }
    if (A_shape[0] == 0 || A_shape[1] == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }
    if (diagonal <= -1 * A_shape[0] || diagonal >= A_shape[1]) {
        utility::LogError(
                "Diagonal parameter must be between [-{}, {}] for a matrix "
                "with shape {}, but got {}.",
                A_shape[0], A_shape[1], A.GetShape().ToString(), diagonal);
    }
}

void Triu(const Tensor& A, Tensor& output, const int diagonal) {
    CheckInput(A, diagonal);
    core::Device device = A.GetDevice();
    output = core::Tensor::Zeros(A.GetShape(), A.GetDtype(), device);
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        TriuCUDA(A.Contiguous(), output, diagonal);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        TriuCPU(A.Contiguous(), output, diagonal);
    }
}

void Tril(const Tensor& A, Tensor& output, const int diagonal) {
    CheckInput(A, diagonal);
    core::Device device = A.GetDevice();
    output = core::Tensor::Zeros(A.GetShape(), A.GetDtype(), device);
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        TrilCUDA(A.Contiguous(), output, diagonal);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        TrilCPU(A.Contiguous(), output, diagonal);
    }
}

void Triul(const Tensor& A, Tensor& upper, Tensor& lower, const int diagonal) {
    CheckInput(A, diagonal);
    core::Device device = A.GetDevice();
    upper = core::Tensor::Zeros(A.GetShape(), A.GetDtype(), device);
    lower = core::Tensor::Zeros(A.GetShape(), A.GetDtype(), device);
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        TriulCUDA(A.Contiguous(), upper, lower, diagonal);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        TriulCPU(A.Contiguous(), upper, lower, diagonal);
    }
}

}  // namespace core
}  // namespace open3d
