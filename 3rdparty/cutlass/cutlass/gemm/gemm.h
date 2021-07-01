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
/*! \file
    \brief Implements a software-pipelined efficient GEMM.
*/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

#include "cutlass/coord.h"
#include "cutlass/util/platform.h"
namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM kernel with launch bounds specified
template <typename Gemm_>
__global__  __launch_bounds__(Gemm_::kThreads)
void gemm_kernel(typename Gemm_::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Gemm_::SharedStorage *shared_storage = 
    reinterpret_cast<typename Gemm_::SharedStorage *>(GemmSharedStorageBase);

  // Construct the GEMM object.
  Gemm_ gemm(params, *shared_storage);

  // Run GEMM.
  gemm.multiply_add();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM kernel without launch bounds specified
template <typename Gemm_>
__global__ /* __launch_bounds__(Gemm_::kThreads) */
void gemm_kernel_nolb(typename Gemm_::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int GemmSharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Gemm_::SharedStorage *shared_storage = 
    reinterpret_cast<typename Gemm_::SharedStorage *>(GemmSharedStorageBase);

  // Construct the GEMM object.
  Gemm_ gemm(params, *shared_storage);

  // Run GEMM.
  gemm.multiply_add();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
/// Partial specialization for launching the GEMM kernel with or without launch bounds
template <typename Gemm, bool WithLaunchBounds>
struct Launch {
  Launch(typename Gemm::Params params, dim3 grid, dim3 block, cudaStream_t stream = 0) {

    int smem_size = int(sizeof(typename Gemm::SharedStorage));
    if (smem_size >= (48 << 10)) {

      cudaError_t result = cudaFuncSetAttribute(
        gemm_kernel<Gemm>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
      );

      if (result != cudaSuccess) {
        return;
      }

      result = cudaFuncSetAttribute(
        gemm_kernel_nolb<Gemm>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

      if (result != cudaSuccess) {
        return; 
      }
    }

    gemm_kernel<Gemm><<< grid, block, sizeof(typename Gemm::SharedStorage), stream >>>(params);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for launching the GEMM kernel with or without launch bounds
template <typename Gemm>
struct Launch<Gemm, false> {
  Launch(typename Gemm::Params params, dim3 grid, dim3 block, cudaStream_t stream = 0) {
    int smem_size = int(sizeof(typename Gemm::SharedStorage));
    if (smem_size >= (48 << 10)) {

      cudaError_t result = cudaFuncSetAttribute(
        gemm_kernel_nolb<Gemm>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
      );

      if (result != cudaSuccess) {
        return;
      }

      result = cudaFuncSetAttribute(
        gemm_kernel_nolb<Gemm>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

      if (result != cudaSuccess) {
        // throw exception?
        return; 
      }
    }

    gemm_kernel_nolb<Gemm><<<
      grid, 
      block, 
      smem_size,
      stream >>>(params);
  }

  // Use device API to launch kernel
  Launch(cudaError_t &result, CUfunction kernel,
         typename Gemm::Params params, dim3 grid, dim3 block, CUstream stream = CU_STREAM_LEGACY) {
    void* params_[] = {const_cast<void*>(reinterpret_cast<void const*>(&params))};

    int smem_size = int(sizeof(typename Gemm::SharedStorage));
    if (smem_size >= (48 << 10)) {

      result = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
      );

      if (result != cudaSuccess) {
        return;
      }

      result = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

      if (result != cudaSuccess) {
        return;
      }
    }

    CUresult launch_result = cuLaunchKernel(
        kernel,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        smem_size, stream, params_, 0);

    if (launch_result != CUDA_SUCCESS) {
      result = cudaErrorLaunchFailure;
      return;
    }

    result = cudaSuccess;
    return;
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_>
struct Gemm {

  /// The traits.
  typedef Traits_ Traits;

  /// Use the params object defined in traits
  typedef typename Traits::Params Params;

  typedef typename Traits::KernelClass KernelClass;

//
// Static function members
//

/// Support for NVRTC
#if !defined(__CUDACC_RTC__)
  /// Launch the kernel.
  static __host__ cudaError_t launch(Params const& params,
                                     cudaStream_t stream = cudaStreamDefault) {

    // Launch the kernel.
    Launch<KernelClass, Traits::GemmConfig::kLaunchBounds>(
      params, params.grid, params.block, stream);

    return cudaGetLastError();
  }

  /// Launch the kernel.
  static __host__ cudaError_t launch(CUfunction kernel,
                                     Params const& params,
                                     CUstream stream = CU_STREAM_LEGACY) {
    cudaError_t result;

    // Launch the kernel.
    Launch<KernelClass, Traits::GemmConfig::kLaunchBounds>(
      result, kernel, params, params.grid, params.block, stream);

    return result;
  }

#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
