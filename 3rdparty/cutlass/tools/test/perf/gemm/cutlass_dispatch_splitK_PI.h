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
#pragma once

#include "cutlass/matrix_traits.h"
#include "tools/util/type_traits.h"
#include <cuda_runtime_api.h>
#include <assert.h>

namespace perf {

template <typename KernelClass_,
    typename Index_,
    typename ScalarA_,
    typename ScalarB_,
    typename ScalarC_,
    typename ScalarD_,
    typename Compute_,
    typename ScalarEpilogue_,
    bool ThreadMultiplyAdd_,
    #if CUTLASS_ENABLE_CUBLAS
    bool RunCuBLAS_ = true
    #else
    bool RunCuBLAS_ = false
    #endif
>
struct CutlassDispatchSplitKPIGemm {
  typedef typename KernelClass_::Params Params;
  typedef KernelClass_ KernelClass;
  typedef Index_ Index;
  typedef ScalarA_ ScalarA;
  typedef ScalarB_ ScalarB;
  typedef ScalarC_ ScalarC;
  typedef ScalarD_ ScalarD;
  typedef Compute_ Compute;
  typedef ScalarEpilogue_ ScalarEpilogue;

  static bool const kThreadMultiplyAdd = ThreadMultiplyAdd_;
  static bool const kRunCuBLAS = RunCuBLAS_;

  static cutlass::MatrixLayout::Kind const kLayoutA = KernelClass::Traits::kLayoutA;
  static cutlass::MatrixLayout::Kind const kLayoutB = KernelClass::Traits::kLayoutB;

  //
  // Data members
  //

  /// Params argument
  Params params;

  /// splitK PI require workspace
  typename cutlass::TypeTraits<Compute>::device_type *workspace_ptr;

  //
  // Methods
  //

  /// Ctor Initializes params object
  CutlassDispatchSplitKPIGemm(Index m,
    Index n,
    Index k,
    ScalarEpilogue alpha,
    ScalarA const* d_a,
    Index lda,
    ScalarB const* d_b,
    Index ldb,
    ScalarEpilogue beta,
    ScalarC const* d_c,
    Index ldc,
    ScalarD* d_d,
    Index ldd) {
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
      std::cout << "reqested workspace memory size("<< workspace_size_in_byte << 
                   ") is larger than available memory size("<< available_device_memory_in_byte << "). Abort." << std::endl;
      throw std::runtime_error("reqested workspace memory size is larger than available memory size. Abort.");
    }

    cudaError_t workspace_err = cudaMalloc(&workspace_ptr, workspace_size_in_byte);
    if (workspace_err != cudaSuccess) {
      std::cout << "\nCUDA workspace malloc error: " << cudaGetErrorString(workspace_err)
        << "\n";
    }

    params.initialize(alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, d_d, ldd, workspace_ptr);
  }

  /// Initializes batched strided params object
  CutlassDispatchSplitKPIGemm(Index m,
    Index n,
    Index k,
    ScalarEpilogue alpha,
    ScalarA const* d_a,
    Index lda,
    long long int batch_stride_A,
    ScalarB const* d_b,
    Index ldb,
    long long int batch_stride_B,
    ScalarEpilogue beta,
    ScalarC const* d_c,
    Index ldc,
    long long int batch_stride_C,
    ScalarD* d_d,
    Index ldd,
    long long int batch_stride_D,
    Index batch_count) {
    assert(0);//batched strided splitK should never be called
  }

  /// Launches kernel
  cudaError_t operator()() { return KernelClass::launch(params); }

  ~CutlassDispatchSplitKPIGemm() {
    cudaError_t workspace_err = cudaFree(workspace_ptr);
    if (workspace_err != cudaSuccess) {
      std::cout << "\nCUDA workspace malloc error: " << cudaGetErrorString(workspace_err)
        << "\n";
    }
  }
};

template<
  typename SplitKPIGemmTraits_
>
struct CutlassDispatchSplitKPIGemmBasic {
  ///
  typedef SplitKPIGemmTraits_ Traits;

  ///
  typedef typename Traits::KernelClass KernelClass;

  /// Index type
  typedef typename Traits::Index Index;

  /// The scalar for A.
  typedef typename Traits::ScalarA ScalarA;
  /// The scalar for B.
  typedef typename Traits::ScalarB ScalarB;
  /// The scalar for C.
  typedef typename Traits::ScalarC ScalarC;
  /// The scalar for D.
  typedef typename Traits::ScalarD ScalarD;
  typedef ScalarD Compute;
  typedef Compute ScalarEpilogue;

  typedef CutlassDispatchSplitKPIGemm<KernelClass,
    Index,
    ScalarA,
    ScalarB,
    ScalarC,
    ScalarD,
    Compute,
    ScalarEpilogue,
    true>
  Dispatch;
};
} //namespace perf
