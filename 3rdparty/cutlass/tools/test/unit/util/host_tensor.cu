/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file

  \brief Defines unit tests for HostTensor and HostMatrix.

  HostTensor is a utility class for allocating memory on the host and on the selected CUDA device
  and presenting a TensorView of this memory.

  HostMatrix is new in CUTLASS 1.1 that offers a matrix-like interface to a HostTensor with rank 2.
  Several examples are shown in this source file.
*/

#include "cutlass_unit_test.h"

#include "cutlass/matrix_traits.h"

#include "tools/util/tensor_view_io.h"
#include "tools/util/host_tensor.h"
#include "tools/util/host_matrix.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

/// Kernel to compute a thread's unique coordinate within a CUDA kernel grid and write a value
/// using a CUTLASS TensorView.
template <typename TensorView>
__global__ void fill_sequential(TensorView view) {

  // Compute the thread's coordinate in the 2D CUDA kernel grid
  cutlass::Coord<2> coord = cutlass::make_Coord(
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y
  );

  // Write a value into the view
  if (view.contains(coord)) {
    view.at(coord) = coord[0] + view.size(0) * coord[1];
  }
}

} // namespace test

////////////////////////////////////////////////////////////////////////////////////////////////////

// This test constructs a CUTLASS HostTensor  with column-major layout.
TEST(HostTensor, fill_sequential_column_major) {

  int const M = 16;
  int const N = 32;

  cutlass::Coord<2> bounds = cutlass::make_Coord(M, N);

  // Construct a rank=2 host tensor of size M-by-N with leading dimension M
  cutlass::HostTensor<
    int,
    2,
    cutlass::MatrixLayout::ColumnMajor> host_tensor(cutlass::make_Coord(M, 1), bounds);

  // Fill it with zeros and synchronize device
  host_tensor.fill(0);
  host_tensor.sync_device();

  // Launch a CUDA kernel by obtaining a TensorView of the device memory
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  test::fill_sequential<<< grid, block >>>(host_tensor.device_view());

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Synchronize the host data
  host_tensor.sync_host();

  // Verify host_tensor contains sequential elements
  int errors = 0;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      int expected = m + n * M;
      int got = host_tensor.at(cutlass::make_Coord(m, n));
      if (expected != got) {
        ++errors;
      }
    }
  }

  EXPECT_EQ(errors, 0) << std::setw(4) << host_tensor << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This test constructs a CUTLASS HostTensor with column-major interleaved layout
TEST(HostTensor, fill_sequential_column_major_interleaved) {

  int const M = 16;
  int const N = 16;
  int const kInterleave = 4;

  cutlass::Coord<2> bounds = cutlass::make_Coord(M, N);

  // Define a mapping function for column-major interleaved layout
  typedef cutlass::MatrixLayout::ColumnMajorInterleaved<kInterleave> TensorRefMapFunc;

  // Construct a rank=2 host tensor of size M-by-N
  cutlass::HostTensor<
    int,
    2,
    TensorRefMapFunc > host_tensor(TensorRefMapFunc::stride(M), bounds);

  // Fill it with zeros and synchronize device
  host_tensor.fill(0);
  host_tensor.sync_device();

  // Launch a CUDA kernel by obtaining a TensorView of the device memory
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  test::fill_sequential<<< grid, block >>>(host_tensor.device_view());

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Synchronize the host data
  host_tensor.sync_host();

  // Verify host_tensor contains sequential elements
  int errors = 0;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      int expected = m + n * M;
      int got = host_tensor.at(cutlass::make_Coord(m, n));
      if (got != expected) {
        ++errors;
      }
    }
  }

  EXPECT_EQ(errors, 0) << std::setw(4) << host_tensor << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// cutlass::HostMatrix extends cutlass::HostTensor of rank=2 to facilitate allocate and operating
// on matrices in device memory.
//
// cutlass::HostMatrix<T> accommodates both row-major and column-major matrices with a single
// leading dimension.
//
// The first test demonstrates use of HostMatrix<> in the same circumstances as HostTensor but with
// simplifcations to the calling interface.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// This test constructs a CUTLASS cutlass::HostMatrix  with column-major layout.
TEST(HostMatrix, fill_sequential_column_major) {

  int const M = 16;
  int const N = 32;
  int const ldm = M + 2; // define leading dimension with padding

  cutlass::Coord<2> bounds = cutlass::make_Coord(M, N);

  // Construct a HostMatrix of size M-by-N with leading dimension ldm
  cutlass::HostMatrix<int> host_matrix(bounds, cutlass::MatrixLayout::kColumnMajor, ldm);

  // Fill it with zeros and synchronize device
  host_matrix.fill(0);
  host_matrix.sync_device();

  // Launch a CUDA kernel by obtaining a TensorView of the device memory
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  test::fill_sequential<<< grid, block >>>(host_matrix.device_view());

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Synchronize the host data
  host_matrix.sync_host();

  // Verify host_matrix contains sequential elements
  int errors = 0;
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      int expected = m + n * M;
      int got = host_matrix.at(cutlass::make_Coord(m, n));
      if (expected != got) {
        ++errors;
      }
    }
  }

  EXPECT_EQ(errors, 0) << std::setw(4) << host_matrix << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Previously, cutlass::HostTensorView<> offered a gemm() method defined for the H and W dimensions.
// The other dimensions were ignored.
//
// To improve the interface, we We have moved this into the HostMatrixView<> and HostMatrix<>
// classes which require rank=2. To accommodate matrix operands of differing layout, we have extracted
// the host-side GEMM implementation into cutlass::reference::host::Gemm() which can compute the
// general matrix product of matrices with arbitrary layout.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// This test constructs a CUTLASS cutlass::HostMatrix  with column-major layout.
TEST(HostMatrix, gemm) {

  // Problem size intentionally small, as reference check has complexity O(MNK).
  int const M = 32;
  int const N = 16;
  int const K = 4;

  int const lda = M;
  int const ldb = N;
  int const ldc = M;

  // Construct matrix operands
  cutlass::HostMatrix<int> A(cutlass::make_Coord(M, K), cutlass::MatrixLayout::kColumnMajor, lda);
  cutlass::HostMatrix<int> B(cutlass::make_Coord(K, N), cutlass::MatrixLayout::kRowMajor, ldb);
  cutlass::HostMatrix<int> C(cutlass::make_Coord(M, N), cutlass::MatrixLayout::kColumnMajor, ldc);

  A.fill_sequential();
  B.fill_sequential();
  C.fill(0);

  int alpha = 1;

  // Compute host-side GEMM reference
  cutlass::reference::host::Gemm(
    cutlass::gemm::GemmCoord(K, N, M),
    alpha,
    A.host_ref(),
    B.host_ref(),
    int(0), // beta
    C.host_ref());

  // Verify result
  int errors = 0;

  // Primitive reference implementation for matrix product
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int result = 0;
      for (int k = 0; k < K; ++k) {
        result += A.at(cutlass::make_Coord(i, k)) * B.at(cutlass::make_Coord(k, j));
      }
      if (C.at(cutlass::make_Coord(i, j)) != alpha * result) {
        ++errors;
      }
    }
  }

  EXPECT_EQ(errors, 0) << "GEMM error\n"
    << "A =\n" << A << "\nB = \n" << B << "\nC =\n" << C << "\n";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// When layout is known at compile time, we may be use the corresponding helper classes to smplify
// matrix instantiation. The matrix layout becomes part of the type which reduces the StorageRank
// of the internal stride vector.
//
// Apart from specifying the matrix layout at compile time, this test is functionally identical to
// HostMatrix.gemm.
//
TEST(HostMatrix, gemm_compile_time_layout) {

  // Problem size intentionally small, as reference check has complexity O(MNK).
  int const M = 32;
  int const N = 16;
  int const K = 4;

  int const lda = M;
  int const ldb = N;
  int const ldc = M;

  // Construct matrix operands
  cutlass::HostMatrixColumnMajor<int> A(cutlass::make_Coord(M, K), lda);
  cutlass::HostMatrixRowMajor<int>    B(cutlass::make_Coord(K, N), ldb);
  cutlass::HostMatrixColumnMajor<int> C(cutlass::make_Coord(M, N), ldc);

  A.fill_sequential();
  B.fill_sequential();
  C.fill(0);

  int alpha = 1;

  // Compute host-side GEMM reference
  cutlass::reference::host::Gemm(
    cutlass::gemm::GemmCoord(K, N, M),
    alpha,
    A.host_ref(),
    B.host_ref(),
    int(0), // beta
    C.host_ref());

  // Verify result
  int errors = 0;

  // Primitive reference implementation for matrix product
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int result = 0;
      for (int k = 0; k < K; ++k) {
        result += A.at(cutlass::make_Coord(i, k)) * B.at(cutlass::make_Coord(k, j));
      }
      if (C.at(cutlass::make_Coord(i, j)) != alpha * result) {
        ++errors;
      }
    }
  }

  EXPECT_EQ(errors, 0) << "GEMM error\n"
    << "A =\n" << A << "\nB = \n" << B << "\nC =\n" << C << "\n";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
