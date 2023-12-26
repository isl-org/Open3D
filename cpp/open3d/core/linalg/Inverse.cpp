// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/Inverse.h"

#include <unordered_map>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"

namespace open3d {
namespace core {

void Inverse(const Tensor &A, Tensor &output) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions
    SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D.", A_shape.size());
    }
    if (A_shape[0] != A_shape[1]) {
        utility::LogError("Tensor must be square, but got {} x {}.", A_shape[0],
                          A_shape[1]);
    }

    int64_t n = A_shape[0];
    if (n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        Tensor ipiv = Tensor::Zeros({n}, core::Int32, device);
        void *ipiv_data = ipiv.GetDataPtr();

        // cuSolver does not support getri, so we have to provide an identity
        // matrix. This matrix is modified in-place as output.
        Tensor A_T = A.T().Contiguous();
        void *A_data = A_T.GetDataPtr();

        output = Tensor::Eye(n, dtype, device);
        void *output_data = output.GetDataPtr();

        InverseCUDA(A_data, ipiv_data, output_data, n, dtype, device);
        output = output.T();
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        Dtype ipiv_dtype;
        if (sizeof(OPEN3D_CPU_LINALG_INT) == 4) {
            ipiv_dtype = core::Int32;
        } else if (sizeof(OPEN3D_CPU_LINALG_INT) == 8) {
            ipiv_dtype = core::Int64;
        } else {
            utility::LogError("Unsupported OPEN3D_CPU_LINALG_INT type.");
        }
        Tensor ipiv = Tensor::Empty({n}, ipiv_dtype, device);
        void *ipiv_data = ipiv.GetDataPtr();

        // LAPACKE supports getri, A is in-place modified as output.
        Tensor A_T = A.T().To(device, /*copy=*/true);
        void *A_data = A_T.GetDataPtr();

        InverseCPU(A_data, ipiv_data, nullptr, n, dtype, device);
        output = A_T.T();
    }
}
}  // namespace core
}  // namespace open3d
