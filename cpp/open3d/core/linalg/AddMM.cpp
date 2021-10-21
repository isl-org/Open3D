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

#include "open3d/core/linalg/AddMM.h"

#include <unordered_map>

namespace open3d {
namespace core {

template <class T>
void AddMM(const Tensor& A, const Tensor& B, Tensor& output, T alpha, T beta) {
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

    if (dtype != core::Float32 && dtype != core::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype.ToString());
        dtype = core::Float32;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();
    bool transA = false;
    bool transB = true;
    bool transC = false;

    int aM = transA ? A.GetShape(1) : A.GetShape(0);
    int aK = transA ? A.GetShape(0) : A.GetShape(1);

    int bK = transB ? B.GetShape(1) : B.GetShape(0);
    int bN = transB ? B.GetShape(0) : B.GetShape(1);

    // output = Tensor::Empty({aM, bN}, dtype, device);

    int cM = transC ? output.GetShape(1) : output.GetShape(0);
    int cN = transC ? output.GetShape(0) : output.GetShape(1);

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Tensor B must be 1D (vector) or 2D (matrix), but got {}D.",
                B_shape.size());
    }
    if (aM != cM) {
        utility::LogError("aM != cM");
    }
    if (aK != bK) {
        utility::LogError("aK != bK");
    }
    if (bN != cN) {
        utility::LogError("bN != cN");
    }

    // Dispatch to backends
    int m = output.GetShape(1);  // stride 1 size
    int n = output.GetShape(0);  // other size
    int k = transA ? A.GetShape(0) : A.GetShape(1);

    int lda = transC ? A.GetStride(0) : B.GetStride(0);
    int ldb = transC ? B.GetStride(0) : A.GetStride(0);
    int ldc = output.GetStride(0);

    // auto gemmTrA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    // auto gemmTrB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    // if (transC) {
    //     gemmTrA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    //     gemmTrB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    // }

    if (m == 0 || k == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_T = A.To(dtype);
    Tensor B_T = B.To(dtype);
    void* A_data = A_T.GetDataPtr();
    void* B_data = B_T.GetDataPtr();

    void* C_data = output.GetDataPtr();

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        AddMMCUDA(B_data, A_data, C_data, m, k, n, alpha, beta, lda, ldb, ldc);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        AddMMCPU(B_data, A_data, C_data, m, k, n, alpha, beta, lda, ldb, ldc);
    }

    output = output.To(dtype_original);
};

template void AddMM(const Tensor& A,
                    const Tensor& B,
                    Tensor& output,
                    float alpha,
                    float beta);

template void AddMM(const Tensor& A,
                    const Tensor& B,
                    Tensor& output,
                    double alpha,
                    double beta);
}  // namespace core
}  // namespace open3d
