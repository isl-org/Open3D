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
  This example demonstrates several CUTLASS utilities in the context of a mixed-precision
  floating-point matrix product computation.

  These utilities are intended to be useful supporting components for managing tensor and matrix
  memory allocations, initializing and comparing results, and computing reference output.

  CUTLASS utilities are defined in the directory `tools/util`, and definitions appear
  namespace `cutlass::` or an inner namespace therein. Operations in `cutlass::reference::` have
  both host-side and device-side implementations, and the choice to use device-side initialization
  and host-side verification in this example was arbitrary.


  cutlass::half_t

    This is a host-only implementation of a half-precision floating-point type. It requires no
    specialized hardware support from the CPU and emulates arithmetic operations. Device-side code
    should use CUDA's `half` type.


  cutlass::HostMatrix<>

    This template class simplifies the creation of a rank=2 tensor with either a column-major or
    row-major layout in memory.

    This class offers methods device_view() and host_view() to provide TensorView objects for
    device- and host-side memory allocations.


  cutlass::reference::device::TensorInitialize()

    This template function initializes the elements of a tensor according to either a procedural
    definition or a random distribution. The function in namespace `cutlass::reference::device::`
    uses a CUDA kernel to perform this initialization, relying on CURAND to compute random numbers.


  cutlass::reference::host::Gemm()

    This template function computes the general matrix product. This template supports unique
    data types for each matrix operand, the internal accumulation type, and the scalar parameters
    alpha and beta.


  cutlass::reference::host::TensorEquals()

    Compares two tensors of identical rank and returns true if values are bit equivalent.

*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__) >= 530

// CUTLASS includes needed for mixed-precision GEMM kernel
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/fp16_sgemm_traits.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "tools/util/tensor_view_io.h"

// Defines cutlass::HostMatrix<>
#include "tools/util/host_matrix.h"

// Defines cutlass::half_t
#include "tools/util/half.h"

// Defines cutlass::reference::device::TensorInitialize()
#include "tools/util/reference/device/tensor_elementwise.h"

// Defines cutlass::reference::host::TensorEquals()
#include "tools/util/reference/host/tensor_elementwise.h"

// Defines cutlass::reference::host::Gemm()
#include "tools/util/reference/host/gemm.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t Cutlass_FP16_SgemmNN(
  int M,
  int N,
  int K,
  cutlass::half_t alpha,
  half const *A,
  int lda,
  half const *B,
  int ldb,
  cutlass::half_t beta,
  half *C,
  int ldc) {

  // Define a CUTLASS Gemm using mixed-precision floating-point.
  //
  // A, B, C, D are half-precision. Internal accumulation is in single-precision.
  //
  // Note, we use CUDA's `half` type for device-side code including CUTLASS GEMM kernels.
  //
  typedef cutlass::gemm::Fp16SgemmSgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<16, 128, 128>,
    half,   // A type
    half,   // B type
    half,   // C type
    half,   // D type
    half    // Scalar type: alpha, beta
  >
    GemmTraits;

  // Define a CUTLASS GEMM object.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  typename Gemm::Params params;

  int result = params.initialize(
    M,                                      // GEMM M dimension
    N,                                      // GEMM N dimension
    K,                                      // GEMM K dimension
    reinterpret_cast<half const &>(alpha),  // scalar alpha
    A,                                      // matrix A operand
    lda,
    B,                                      // matrix B operand
    ldb,
    reinterpret_cast<half const &>(beta),   // scalar beta
    C,                                      // source matrix C
    ldc,
    C,                                      // destination matrix C (may be different memory than source C matrix)
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

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha, cutlass::half_t beta) {
  cudaError_t result;

  //
  // Construct cutlass::HostMatrix<> using the half-precision host-side type.
  //
  // cutlass::HostMatrix<> allocates memory on both the host and device corresponding to rank=2
  // tensors in column-major layout. Explicit synchronization methods are offered to copy the
  // tensor to the device or to the host.
  //

  // M-by-K matrix of cutlass::half_t
  cutlass::HostMatrix<cutlass::half_t> A(cutlass::MatrixCoord(M, K));

  // K-by-N matrix of cutlass::half_t
  cutlass::HostMatrix<cutlass::half_t> B(cutlass::MatrixCoord(K, N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostMatrix<cutlass::half_t> C_cutlass(cutlass::MatrixCoord(M, N));

  // M-by-N matrix of cutlass::half_t
  cutlass::HostMatrix<cutlass::half_t> C_reference(cutlass::MatrixCoord(M, N));

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

  result = Cutlass_FP16_SgemmNN(
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

  // A and B were initialized using device-side procedures. The intent of this example is to
  // use the host-side reference GEMM, so we must perform a device-to-host copy.
  A.sync_host();
  B.sync_host();

  // Copy CUTLASS's GEMM results into host memory.
  C_cutlass.sync_host();

  // Compute the reference result using the host-side GEMM reference implementation.
  cutlass::reference::host::Gemm(
    cutlass::gemm::GemmCoord(K, N, M),  // problem size  (type: cutlass::gemm::GemmCoord)
    alpha,                              // alpha         (type: cutlass::half_t)
    A.host_ref(),                       // A             (concept: TensorRef)
    B.host_ref(),                       // B             (concept: TensorRef)
    beta,                               // beta          (type: cutlass::half_t)
    C_reference.host_ref(),             // C             (concept: TensorRef)
    float(0)                            // Accumulator initial value passed as argument to deduce
  );                                    // internal accumulation data type as float.

  // Compare reference to computed results.
  if (!cutlass::reference::host::TensorEquals(C_reference.host_view(), C_cutlass.host_view())) {

    std::cerr << "Error - CUTLASS mixed-precision GEMM kernel differs from reference." << std::endl;

    //
    // On error, print C_cutlass and C_reference to std::cerr.
    //
    // Note, these are matrices of half-precision elements stored in host memory as
    // arrays of type cutlass::half_t.
    //

    // Result of CUTLASS mixed-precision GEMM kernel
    std::cerr << "CUTLASS:\n" << C_cutlass << std::endl;

    // Result of reference computation
    std::cerr << "Reference:\n" << C_reference << std::endl;

    // Return error code.
    return cudaErrorUnknown;
  }

  // Passed error check
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to cutlass_utilities example.
//
// usage:
//
//   01_cutlass_utilities <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // This example uses half-precision and is only suitable for devices with compute capabitliy 5.3 or greater.
  //

  cudaDeviceProp prop;
  cudaError_t result = cudaGetDeviceProperties(&prop, 0);
  
  if (result != cudaSuccess) {
    std::cerr << "Failed to query device properties with error " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
    std::cerr << "This example uses mixed precision and is only suitable for devices with compute capability 5.3 or greater.\n";
    std::cerr << "You are using a CUDA device with compute capability " << prop.major << "." << prop.minor << std::endl;
    return -1;
  }

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions: <M> <N> <K>
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Linear scale factors in GEMM. Note, these are half-precision values stored as
  // cutlass::half_t.
  //
  // Values outside the range of IEEE FP16 will overflow to infinity or underflow to zero.
  //
  cutlass::half_t scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);

    ss >> scalars[i - 4];   // lexical cast to cutlass::half_t
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif

