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

/*
  This example demonstrates how to call a CUTLASS GEMM kernel using Turing integer WMMA.

  The CUTLASS integer WMMA Gemm template is instantiated in the function Cutlass_S8_WmmagemmNN. This
  is kernel computes the general matrix product (GEMM) using integer arithmetic accelerated by Turing
  WMMA and assumes all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  This example uses CUTLASS utilities to ease the matrix operations.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes needed for WMMA GEMM kernel
#include "cutlass/wmma_matrix.h"

#pragma warning( disable : 4503)

// This example works only when this MACRO is defined in "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_INT_WMMA

// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::WmmaGemmTraits, the structural components for WMMA GEMM
#include "cutlass/gemm/wmma_gemm_traits.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "tools/util/tensor_view_io.h"

// Defines cutlass::HostMatrix<>
#include "tools/util/host_matrix.h"

// Defines cutlass::reference::device::TensorInitialize()
#include "tools/util/reference/device/tensor_elementwise.h"

// Defines cutlass::reference::host::TensorEquals()
#include "tools/util/reference/host/tensor_elementwise.h"

// Defines cutlass::reference::host::Gemm()
#include "tools/util/reference/host/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t Cutlass_S8_WmmagemmNN(
  int M,
  int N,
  int K,
  int alpha,
  signed char const *A,
  int lda,
  signed char const *B,
  int ldb,
  int beta,
  int *C,
  int ldc) {

  // Define type definition for 8-bit signed int WMMA CUTLASS GEMM with column-major
  // input matrices and 128x128x128 threadblock tile size.
  //
  // Note, A and B are 8-bit signed int. C and D are 32-bit int. .
  //
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
    cutlass::Shape<128, 128, 128>,         // threadblock tile size
    signed char,                           // A type
    signed char,                           // B type
    int,                                   // D type
    cutlass::gemm::LinearScaling<int>,     // functor to do the math in the epilogue
    int,                                   // accumulator type
    cutlass::Shape<128, 32, 32>,           // warp tile size
    cutlass::Shape<16, 16, 16>,            // WMMA instruction tile size
    16,                                    // scalars every time a thread loads from A
    16                                     // scalars every time a thread loads from B
  >
    GemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  typename Gemm::Params params;

  int result = params.initialize(
    M,     // GEMM M dimension
    N,     // GEMM N dimension
    K,     // GEMM K dimension
    alpha, // scalar alpha
    A,     // matrix A operand
    lda,
    B,     // matrix B operand
    ldb,
    beta,  // scalar beta
    C,     // source matrix C
    ldc,
    C,     // destination matrix C (may be different memory than source C matrix)
    ldc
  );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call an integer
/// CUTLASS WMMA GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, int alpha, int beta) {
  cudaError_t result;

  //
  // Construct cutlass::HostMatrix<> using the integer host-side types.

  // M-by-K matrix of signed char
  cutlass::HostMatrix<signed char> A(cutlass::MatrixCoord(M, K));

  // K-by-N matrix of signed char
  cutlass::HostMatrix<signed char> B(cutlass::MatrixCoord(K, N));

  // M-by-N matrix of int
  cutlass::HostMatrix<int> C_cutlass(cutlass::MatrixCoord(M, N));

  // M-by-N matrix of int
  cutlass::HostMatrix<int> C_reference(cutlass::MatrixCoord(M, N));

  //
  // Initialize matrices with small, random integers.
  //

  cutlass::Distribution dist;

  // Uniform random distribution from -4 .. 4. Values are truncated to integers.
  dist.set_uniform(-4, 4);

  // Arbitrary RNG seed value. Hard-coded for deterministic results.
  int seed = 2080;

  cutlass::reference::device::TensorInitialize(
    A.device_view(),                                // concept: TensorView
    seed,
    dist);

  cutlass::reference::device::TensorInitialize(
    B.device_view(),                                // concept: TensorView
    seed * 2,
    dist);

  cutlass::reference::device::TensorInitialize(
    C_cutlass.device_view(),                        // concept: TensorView
    seed * 3,
    dist);

  // Copy C_cutlass into C_reference so the GEMM is correct when beta != 0.
  cutlass::reference::device::TensorFill(C_reference.device_view(), C_cutlass.device_view());

  // Copy the device-side view into host memory
  C_reference.sync_host();

  //
  // Launch the CUTLASS GEMM kernel
  //

  result = Cutlass_S8_WmmagemmNN(
    M,
    N,
    K,
    alpha,
    A.device_data(),
    A.leading_dim(),
    B.device_data(),
    B.leading_dim(),
    beta,
    C_cutlass.device_data(),
    C_cutlass.leading_dim()
  );

  if (result != cudaSuccess) {
    return result;
  }

  //
  // Verify the result using a host-side reference
  //

  // A and B were initialized using device-side procedures.
  A.sync_host();
  B.sync_host();

  // Copy CUTLASS's GEMM results into host memory.
  C_cutlass.sync_host();

  // Compute the reference result using the host-side GEMM reference implementation.
  cutlass::reference::host::Gemm(
    cutlass::gemm::GemmCoord(K, N, M),  // problem size  (type: cutlass::gemm::GemmCoord)
    alpha,                              // alpha         (type: int)
    A.host_ref(),                       // A             (concept: TensorRef)
    B.host_ref(),                       // B             (concept: TensorRef)
    beta,                               // beta          (int)
    C_reference.host_ref(),             // C             (concept: TensorRef)
    int(0)                              // Accumulator initial value passed as argument to deduce
  );                                    // internal accumulation data type as int.

  // Compare reference to computed results.
  if (!cutlass::reference::host::TensorEquals(C_reference.host_view(), C_cutlass.host_view())) {

    std::cerr << "Error - CUTLASS WMMA GEMM kernel differs from reference." << std::endl;

    //
    // On error, print C_cutlass and C_reference to std::cerr.
    //

    // Result of CUTLASS WMMA GEMM kernel
    std::cerr << "CUTLASS:\n" << C_cutlass << std::endl;

    // Result of reference computation
    std::cerr << "Reference:\n" << C_reference << std::endl;

    // Return error code.
    return cudaErrorUnknown;
  }

  // Passed error check
  return cudaSuccess;
}
#endif // defined CUTLASS_USE_INT_WMMA

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to wmma_gemm example.
//
// usage:
//
//   05_wmma_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

#ifdef CUTLASS_USE_INT_WMMA
  // Properties of CUDA device
  cudaDeviceProp device_properties;

  // Assumne the device id is 0.
  int device_id = 0;

  cudaError_t result = cudaGetDeviceProperties(&device_properties, device_id);
  if (result != cudaSuccess) {
    std::cerr << "Failed to get device properties: " 
      << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  if ((device_properties.major * 10 +  device_properties.minor) < 72) {
    std::cerr << "This example needs to run on a device which has at least 7.2 compute capability." << std::endl;
    return -1;
  }

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  int scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;

#else
  std::cerr << "CUTLASS WMMA GEMM targeting Turing Tensor Cores features requires compute capability 7.2." << std::endl;
  return -1;
#endif // defined CUTLASS_USE_INT_WMMA
}

///////////////////////////////////////////////////////////////////////////////////////////////////

