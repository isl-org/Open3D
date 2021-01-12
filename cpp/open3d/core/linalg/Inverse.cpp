// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/core/linalg/Inverse.h"

#include <unordered_map>

#include "open3d/core/linalg/LinalgHeadersCPU.h"

namespace open3d {
namespace core {

void Inverse(const Tensor &A, Tensor &output) {
    // Check devices
    Device device = A.GetDevice();

    // Check dtypes
    Dtype dtype = A.GetDtype();
    if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
        utility::LogError(
                "Only tensors with Float32 or Float64 are supported, but "
                "received {}.",
                dtype.ToString());
    }

    // Check dimensions
    SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }
    if (A_shape[0] != A_shape[1]) {
        utility::LogError("Tensor A must be square, but got {} x {}.",
                          A_shape[0], A_shape[1]);
    }

    int64_t n = A_shape[0];
    if (n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        Tensor ipiv = Tensor::Zeros({n}, Dtype::Int32, device);
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
            ipiv_dtype = Dtype::Int32;
        } else if (sizeof(OPEN3D_CPU_LINALG_INT) == 8) {
            ipiv_dtype = Dtype::Int64;
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
