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

#include "open3d/core/linalg/Matmul.h"

#include <unordered_map>

#include "open3d/core/linalg/MatmulImp.h"

namespace open3d {
namespace core {

void Matmul(const Tensor& A, const Tensor& B, Tensor& output) {
    // Check devices
    Device device = A.GetDevice();
    if (device != B.GetDevice()) {
        utility::LogError("Tensor A device {} and Tensor B device {} mismatch.",
                          A.GetDevice().ToString(), B.GetDevice().ToString());
    }

    // Check dtypes
    Dtype dtype = A.GetDtype(), dtype_original = dtype;
    if (dtype != B.GetDtype()) {
        utility::LogError("Tensor A dtype {} and Tensor B dtype {} mismatch.",
                          A.GetDtype().ToString(), B.GetDtype().ToString());
    }

    if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype.ToString());
        dtype = Dtype::Float32;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Tensor B must be 1D (vector) or 2D (matrix), but got {}D.",
                B_shape.size());
    }
    if (A_shape[1] != B_shape[0]) {
        utility::LogError("Tensor A columns {} mismatch with Tensor B rows {}.",
                          A_shape[1], B_shape[0]);
    }

    // Dispatch to backends
    int64_t m = A_shape[0];
    int64_t k = A_shape[1];
    int64_t n = B_shape.size() == 2 ? B_shape[1] : 1;

    if (m == 0 || k == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_T = A.T().Contiguous().To(dtype);
    Tensor B_T = B.T().Contiguous().To(dtype);
    void* A_data = A_T.GetDataPtr();
    void* B_data = B_T.GetDataPtr();

    output = Tensor::Empty({n, m}, dtype, device);
    void* C_data = output.GetDataPtr();

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        MatmulCUDA(A_data, B_data, C_data, m, k, n, dtype);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        MatmulCPU(A_data, B_data, C_data, m, k, n, dtype);
    }

    output = output.T().To(dtype_original);
};

}  // namespace core
}  // namespace open3d
