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

#include "open3d/core/linalg/SVD.h"

#include <unordered_map>

#include "open3d/core/linalg/performance/Matrix.h"
#include "open3d/core/linalg/performance/SVD3x3.h"

namespace open3d {
namespace core {

void SVD(const Tensor &A, Tensor &U, Tensor &S, Tensor &VT) {
    // Check dtypes
    Dtype dtype = A.GetDtype();
    if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
        utility::LogError(
                "Only tensors with Float32 or Float64 are supported, but "
                "received {}",
                dtype.ToString());
    }

    // Check dimensions
    SizeVector A_shape = A.GetShape();

    core::Tensor U_3x3;
    core::Tensor S_3x1;
    core::Tensor VT_3x3;

    // Check devices
    Device device = A.GetDevice();

    if (A.GetShape() == open3d::core::SizeVector({3, 3})) {
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            U_3x3 = core::Tensor::Empty({3, 3}, dtype, core::HostDevice);
            S_3x1 = core::Tensor::Empty({3, 1}, dtype, core::HostDevice);
            VT_3x3 = core::Tensor::Empty({3, 3}, dtype, core::HostDevice);

            scalar_t *U_3x3_ptr = U_3x3.GetDataPtr<scalar_t>();
            scalar_t *S_3x1_ptr = S_3x1.GetDataPtr<scalar_t>();
            scalar_t *VT_3x3_ptr = VT_3x3.GetDataPtr<scalar_t>();

            core::Tensor A_3x3 = A.To(core::HostDevice, false).Contiguous();
            const scalar_t *A_3x3_ptr = A_3x3.GetDataPtr<scalar_t>();

            svd3x3(A_3x3_ptr, U_3x3_ptr, S_3x1_ptr, VT_3x3_ptr);
            transpose3x3_(VT_3x3_ptr);
        });

        U = U_3x3.To(device);
        S = S_3x1.To(device);
        VT = VT_3x3.To(device);
        return;

    } else {
        if (A_shape.size() != 2) {
            utility::LogError("Tensor must be 2D, but got {}D", A_shape.size());
        }

        int64_t m = A_shape[0], n = A_shape[1];
        if (m == 0 || n == 0) {
            utility::LogError(
                    "Tensor shapes should not contain dimensions with zero.");
        }
        if (m < n) {
            utility::LogError("Only support m >= n, but got {} and {} matrix",
                              m, n);
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

        if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
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

    return;
}

}  // namespace core
}  // namespace open3d
