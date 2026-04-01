// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/Gramian.h"

#include <unordered_map>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

void Gram(const Tensor& A, Tensor& output) {
    const Device device = A.GetDevice();
    const Dtype dtype_original = A.GetDtype();
    Dtype dtype;

    if (dtype_original != core::Float32 && dtype_original != core::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype_original.ToString());
        dtype = core::Float32;
    } else {
        dtype = dtype_original;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }

    // Dispatch to backends
    int64_t m = A_shape[0];
    int64_t n = A_shape[1];

    if (m == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_contiguous = A.Contiguous().To(dtype);
    void* A_data = A_contiguous.GetDataPtr();

    output = Tensor::Empty({n, n}, dtype, device);
    void* B_data = output.GetDataPtr();

    if (device.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        GramSYCL(A_data, B_data, m, n, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        GramCUDA(A_data, B_data, m, n, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        GramCPU(A_data, B_data, m, n, dtype);
    }

    output = output.To(dtype_original);
}

void RowGram(const Tensor& A, Tensor& output) {
    const Device device = A.GetDevice();
    const Dtype dtype_original = A.GetDtype();
    Dtype dtype;

    if (dtype_original != core::Float32 && dtype_original != core::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype_original.ToString());
        dtype = core::Float32;
    } else {
        dtype = dtype_original;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }

    // Dispatch to backends
    int64_t m = A_shape[0];
    int64_t n = A_shape[1];

    if (m == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_contiguous = A.Contiguous().To(dtype);
    void* A_data = A_contiguous.GetDataPtr();

    output = Tensor::Empty({m, m}, dtype, device);
    void* B_data = output.GetDataPtr();

    if (device.IsSYCL()) {
#ifdef BUILD_SYCL_MODULE
        RowGramSYCL(A_data, B_data, m, n, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        RowGramCUDA(A_data, B_data, m, n, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        RowGramCPU(A_data, B_data, m, n, dtype);
    }

    output = output.To(dtype_original);
}

}  // namespace core
}  // namespace open3d
