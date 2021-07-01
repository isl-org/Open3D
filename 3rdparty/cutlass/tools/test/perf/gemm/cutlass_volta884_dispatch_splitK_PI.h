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


////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once


#include "cutlass/gemm/device_gemm.h"
#include "cutlass/gemm/volta884_gemm_traits.h"

#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
struct Volta884GemmDispatchSplitKPI {

  typedef cutlass::gemm::DeviceGemm<Traits> Gemm;

  typedef typename Gemm::Params Params;

  typedef typename Traits::ScalarC ScalarC;
  typedef typename Traits::ScalarD ScalarD;
  typedef typename Traits::Scalar ScalarEpilogue;

  /// Indicate warp-level GEMM
  static bool const kThreadMultiplyAdd = false;

  #if CUTLASS_ENABLE_CUBLAS
  static bool const kRunCuBLAS = true;
  #else
  static bool const kRunCuBLAS = false;
  #endif

  static cutlass::MatrixLayout::Kind const kLayoutA = Traits::kLayoutA;
  static cutlass::MatrixLayout::Kind const kLayoutB = Traits::kLayoutB;

  //
  // Data members
  //

  /// Params argument
  Params params;

  /// splitK PI require workspace
  typename cutlass::TypeTraits<typename Traits::ScalarAccum>::device_type *workspace_ptr;

  //
  // Methods
  //

  Volta884GemmDispatchSplitKPI() {}

  /// Initializes params object
  Volta884GemmDispatchSplitKPI(int m, int n, int k, ScalarEpilogue alpha, half const* d_a, int lda,
                  half const* d_b, int ldb, ScalarEpilogue beta, ScalarC const* d_c, int ldc,
                  ScalarD* d_d, int ldd) {
    params.init_problem(m, n, k);
    size_t workspace_size_in_byte = params.required_workspace_memory_in_byte();
    size_t available_device_memory_in_byte = 0;
    size_t device_memory_in_byte = 0;
    cudaError_t cudaMemGetInfo_err = cudaMemGetInfo(&available_device_memory_in_byte, &device_memory_in_byte);
    if (cudaMemGetInfo_err != cudaSuccess) {
      std::cout << "\ncudaMemGetInfo error: " << cudaGetErrorString(cudaMemGetInfo_err)
        << "\n";
    }

    if (workspace_size_in_byte > available_device_memory_in_byte) {
      std::cout << "reqested workspace memory size(" << workspace_size_in_byte <<
        ") is larger than available memory size(" << available_device_memory_in_byte << "). Abort." << std::endl;
      throw std::runtime_error("reqested workspace memory size is larger than available memory size. Abort.");
    }

    cudaError_t workspace_err = cudaMalloc(&workspace_ptr, workspace_size_in_byte);
    if (workspace_err != cudaSuccess) {
      std::cout << "\nCUDA workspace malloc error: " << cudaGetErrorString(workspace_err)
        << "\n";
    }

    params.initialize(alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, d_d, ldd, workspace_ptr, 8 /*volta884 requires leading dim to be mulitiple of 8*/);
  }

  Volta884GemmDispatchSplitKPI(int m,
    int n,
    int k,
    ScalarEpilogue alpha,
    half const* d_a,
    int lda,
    long long int batch_stride_A,
    half const* d_b,
    int ldb,
    long long int batch_stride_B,
    ScalarEpilogue beta,
    ScalarC const* d_c,
    int ldc,
    long long int batch_stride_C,
    ScalarD* d_d,
    int ldd,
    long long int batch_stride_D,
    int batch_count) {
    assert(0);//not yet supported
  }

  /// Initializes params object
  Volta884GemmDispatchSplitKPI(Params const& _params) : params(_params) {}

  /// Launches kernel
  cudaError_t operator()() { 
    return Gemm::launch(params); 
  }
};
