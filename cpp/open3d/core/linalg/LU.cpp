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

#include "open3d/core/linalg/LU.h"

#include "open3d/core/linalg/LUImpl.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/Tri.h"

namespace open3d {
namespace core {

// Get column permutation tensor from ipiv (swaping index array).
static core::Tensor GetColPermutation(const Tensor& ipiv,
                                      int number_of_indices,
                                      int number_of_rows) {
    Tensor full_ipiv =
            Tensor::Arange(0, number_of_rows, 1, core::Int32, Device("CPU:0"));
    Tensor ipiv_cpu = ipiv.To(Device("CPU:0"), core::Int32, /*copy=*/false);
    const int* ipiv_ptr = static_cast<const int*>(ipiv_cpu.GetDataPtr());
    int* full_ipiv_ptr = static_cast<int*>(full_ipiv.GetDataPtr());
    for (int i = 0; i < number_of_indices; i++) {
        int temp = full_ipiv_ptr[i];
        full_ipiv_ptr[i] = full_ipiv_ptr[ipiv_ptr[i] - 1];
        full_ipiv_ptr[ipiv_ptr[i] - 1] = temp;
    }
    // This is column permutation for P, where P.A = L.U.
    // Int64 is required by AdvancedIndexing.
    return full_ipiv.To(ipiv.GetDevice(), core::Int64, /*copy=*/false);
}

// Decompose output in P, L, U matrix form.
static void OutputToPLU(const Tensor& output,
                        Tensor& permutation,
                        Tensor& lower,
                        Tensor& upper,
                        const Tensor& ipiv,
                        const bool permute_l) {
    int n = output.GetShape()[0];
    core::Device device = output.GetDevice();

    // Get upper and lower matrix from output matrix.
    Triul(output, upper, lower, 0);
    // Get column permutaion vector from pivot indices vector.
    Tensor col_permutation = GetColPermutation(ipiv, ipiv.GetShape()[0], n);
    // Creating "Permutation Matrix (P in P.A = L.U)".
    permutation = core::Tensor::Eye(n, output.GetDtype(), device)
                          .IndexGet({col_permutation});
    // Calculating P in A = P.L.U. [P.Inverse() = P.T()].
    permutation = permutation.T().Contiguous();
    // Permute_l option, to return L as L = P.L.
    if (permute_l) {
        lower = permutation.Matmul(lower);
    }
}

void LUIpiv(const Tensor& A, Tensor& ipiv, Tensor& output) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions.
    const SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D.", A_shape.size());
    }

    const int64_t rows = A_shape[0];
    const int64_t cols = A_shape[1];
    if (rows == 0 || cols == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    // "output" tensor is modified in-place as ouput.
    // Operations are COL_MAJOR.
    output = A.T().Clone();
    void* A_data = output.GetDataPtr();

    // Returns LU decomposition in form of an output matrix,
    // with lower triangular elements as L, upper triangular and diagonal
    // elements as U, (diagonal elements of L are unity), and ipiv array,
    // which has the pivot indices (for 1 <= i <= min(M,N), row i of the
    // matrix was interchanged with row IPIV(i).
    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        int64_t ipiv_len = std::min(rows, cols);
        ipiv = core::Tensor::Empty({ipiv_len}, core::Int32, device);
        void* ipiv_data = ipiv.GetDataPtr();
        LUCUDA(A_data, ipiv_data, rows, cols, dtype, device);
#else
        utility::LogInfo("Unimplemented device.");
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

        int64_t ipiv_len = std::min(rows, cols);
        ipiv = core::Tensor::Empty({ipiv_len}, ipiv_dtype, device);
        void* ipiv_data = ipiv.GetDataPtr();
        LUCPU(A_data, ipiv_data, rows, cols, dtype, device);
    }
    // COL_MAJOR -> ROW_MAJOR.
    output = output.T().Contiguous();
}

void LU(const Tensor& A,
        Tensor& permutation,
        Tensor& lower,
        Tensor& upper,
        const bool permute_l) {
    AssertTensorDtypes(A, {Float32, Float64});

    // Get output matrix and ipiv.
    core::Tensor ipiv, output;
    LUIpiv(A, ipiv, output);

    // Decompose output in P, L, U matrix form.
    OutputToPLU(output, permutation, lower, upper, ipiv, permute_l);

    // For non-square input case of shape {rows, cols}, shape of P, L, U:
    // P {rows, rows}; L {rows, min(rows, cols)}; U {min(rows, cols), cols}.
    if (A.GetShape()[0] != A.GetShape()[1]) {
        int64_t min_ = std::min(A.GetShape()[0], A.GetShape()[1]);
        lower = lower.Slice(1, 0, min_);
        upper = upper.Slice(0, 0, min_);
    }
}

}  // namespace core
}  // namespace open3d
