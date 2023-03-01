// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/LeastSquares.h"

#include <unordered_map>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

void LeastSquares(const Tensor &A, const Tensor &B, Tensor &X) {
    AssertTensorDtypes(A, {Float32, Float64});
    AssertTensorDevice(B, A.GetDevice());
    AssertTensorDtype(B, A.GetDtype());

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D", A_shape.size());
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Tensor B must be 1D (vector) or 2D (matrix), but got {}D.",
                B_shape.size());
    }
    if (B_shape[0] != A_shape[0]) {
        utility::LogError("Tensor A and B's first dimension mismatch.");
    }

    int64_t m = A_shape[0];
    int64_t n = A_shape[1];
    int64_t k = B_shape.size() == 2 ? B_shape[1] : 1;
    if (m == 0 || n == 0 || k == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    if (m < n) {
        utility::LogError("Tensor A shape must satisfy rows({}) > cols({}).", m,
                          n);
    }

    // A and B are modified in-place
    Tensor A_copy = A.T().Clone();
    Tensor B_copy = B.T().Clone();

    void *A_data = A_copy.GetDataPtr();
    void *B_data = B_copy.GetDataPtr();

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        LeastSquaresCUDA(A_data, B_data, m, n, k, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        LeastSquaresCPU(A_data, B_data, m, n, k, dtype, device);
    }

    X = B_copy.T().Slice(0, 0, n);
}
}  // namespace core
}  // namespace open3d
