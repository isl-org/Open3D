// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/SVD.h"

#include <unordered_map>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

void SVD(const Tensor &A, Tensor &U, Tensor &S, Tensor &VT) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions
    SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D", A_shape.size());
    }

    int64_t m = A_shape[0], n = A_shape[1];
    if (m == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }
    if (m < n) {
        utility::LogError("Only support m >= n, but got {} and {} matrix", m,
                          n);
    }

    Tensor A_T = A.T().Contiguous();
    U = Tensor::Empty({m, m}, dtype, device);
    S = Tensor::Empty({n}, dtype, device);
    VT = Tensor::Empty({n, n}, dtype, device);
    Tensor superb = Tensor::Empty({std::min(m, n) - 1}, dtype, device);

    void *A_data = A_T.GetDataPtr();
    void *U_data = U.GetDataPtr();
    void *S_data = S.GetDataPtr();
    void *VT_data = VT.GetDataPtr();
    void *superb_data = superb.GetDataPtr();

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        SVDCUDA(A_data, U_data, S_data, VT_data, superb_data, m, n, dtype,
                device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        SVDCPU(A_data, U_data, S_data, VT_data, superb_data, m, n, dtype,
               device);
    }
    U = U.T();
    VT = VT.T();
}
}  // namespace core
}  // namespace open3d
