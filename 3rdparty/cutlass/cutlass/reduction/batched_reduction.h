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
\brief Implements a software-pipelined efficient batched reduction.
D = alpha * Reduction(A) + beta * C
*/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

#include "cutlass/coord.h"
#include "cutlass/util/platform.h"
#include "cutlass/fragment.h"

namespace cutlass {
namespace reduction {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename batched_reduction_>
__global__ __launch_bounds__(batched_reduction_::Traits::kThreads, 1) void batched_reduction_kernel(typename batched_reduction_::Params params) {
  // Construct the batched_reduction object
  batched_reduction_ batched_reduction(params);
  batched_reduction.run();
}

template <typename BatchedReductionTraits_>
struct BatchedReduction {
  /// This class
  typedef BatchedReduction<BatchedReductionTraits_> This_;
  /// The traits
  typedef BatchedReductionTraits_ Traits;
  /// Params
  typedef typename Traits::Params Params;
  /// functor
  typedef typename Traits::Functor Functor;

  /// ctor
  CUTLASS_DEVICE BatchedReduction(Params const &params_)
      : params(params_), functor(params_.functorParams) {}
  
  /// main operation method
  /// D = alpha * Reduction(A) + beta * C
  CUTLASS_DEVICE void run() {
#if (__CUDA_ARCH__ >= 600)
    // Swizzle the IDs of the block 
    typename Traits::BlockSwizzle block_swizzle;
    Coord<3> threadblock_offset =
      block_swizzle.get_threadblock_offset(make_Coord_from_shape<typename Traits::SubTile>());

    int subTileSize = gridDim.x * Traits::SubTile::kW;
    int tileSize = params.problem_size[1] * params.problem_size[2];
    int subTileOffset = threadblock_offset[2] + threadIdx.x * Traits::ThreadShape::kW;

    int subTileBase = 0;

    typename Traits::ScalarA inRegs[Traits::maxInReg];
    typename Traits::ScalarAccum AccumRegs[Traits::maxOutReg];
#pragma unroll
    for (int subTile = 0; subTile < tileSize; subTile += subTileSize) {
      int tileOffset = subTileBase + subTileOffset;
      // Init AccumRegs
#pragma unroll
      for (int i = 0; i < Traits::ThreadShape::kW; i++)
        AccumRegs[i] = static_cast<typename Traits::ScalarAccum>(0.0f);
      // Fetch c0
      typename Traits::ScalarAccum c0[Traits::ThreadShape::kW];
#pragma unroll
      for (int i = 0; i< Traits::ThreadShape::kW; i++)
        c0[i] = static_cast<typename Traits::ScalarAccum>(params.d_c[tileOffset + i]);

      // Fetch partial sums from A
#pragma unroll
      for (int s = 0; s < Traits::ReductionSize; s++) {
        int inRegOffset = s * Traits::ThreadShape::kW;
        int dOffset = (s * tileSize) + tileOffset;
#pragma unroll
        for (int i = 0; i< Traits::ThreadShape::kW; i++) {
          inRegs[inRegOffset + i] = params.d_a[dOffset + i];
        }
      }

      // Accumulate
#pragma unroll
      for (int s = 0; s < Traits::ReductionSize; s++) {
        int inRegOffset = s * Traits::ThreadShape::kW;
#pragma unroll
        for (int i = 0; i < Traits::ThreadShape::kW; i++) {
          //AccumRegs[i] = cuFma(params.alpha, inRegs[inRegOffset + i], AccumRegs[i]);
          //AccumRegs[i] = params.alpha * inRegs[inRegOffset + i] + AccumRegs[i];
          AccumRegs[i] = static_cast<typename Traits::ScalarAccum>(inRegs[inRegOffset + i]) + AccumRegs[i];
        }
      }
      // calling functor
      functor_caller<Traits::ThreadShapeMultiple2>(AccumRegs, c0, AccumRegs);

      // Store AccumRegs to D
#pragma unroll
      for (int i = 0; i < Traits::ThreadShape::kW; i++) {
        params.d_d[tileOffset + i] = static_cast<typename Traits::ScalarD>(AccumRegs[i]);
      }

      // Advance sub-tile pointer
      subTileBase += subTileSize;
    } // end for loop
#endif //#if (__CUDA_ARCH__ >= 600)
  }

  template<bool ThreadShapeMultiple2>
  CUTLASS_DEVICE void functor_caller(typename Traits::ScalarAccum const *accum, typename Traits::ScalarAccum const *old, typename Traits::ScalarAccum *output) {
    if (ThreadShapeMultiple2 == true) {
#pragma unroll
      for (int i = 0; i < Traits::ThreadShape::kW / 2; i++) {
        functor.template evaluate<typename Traits::ScalarAccum, typename Traits::ScalarAccum, 2>(&accum[2 * i], &old[2 * i], &output[2 * i]);
      }
    }
    else {
#pragma unroll
      for (int i = 0; i < Traits::ThreadShape::kW; i++) {
        functor.template evaluate<typename Traits::ScalarAccum, typename Traits::ScalarAccum, 1>(&accum[i], &old[i], &output[i]);
      }
    }
  }

  //
  // Static function members
  //
#if !defined(__CUDACC_RTC__)
  /// Launch the kernel.
  static __host__ cudaError_t launch(Params const& params,
    cudaStream_t stream = cudaStreamDefault) {
    // Setup the grid. 
    typename Traits::BlockSwizzle block_swizzle;
    dim3 grid = block_swizzle.get_grid_layout(params.problem_size,
                                              make_Coord_from_shape<typename Traits::OutputTile>());
    
    dim3 block;
    block.x = Traits::kThreads;
    batched_reduction_kernel<This_><<<grid, block, 0, stream>>>(params);
    return cudaGetLastError();
  }
#endif

  //
  // Data members
  //

  /// The params.
  Params const& params;
  // The functor.
  Functor functor;
};

} // namespace reduction
} // namespace cutlass
