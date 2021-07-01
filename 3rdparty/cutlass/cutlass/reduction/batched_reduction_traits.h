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
\brief Defines structural properties of complete batched reduction.
D = alpha * Reduction(A) + beta * C
*/
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/shape.h"
#include "cutlass/reduction/threadblock_swizzle.h"
#include "cutlass/reduction/batched_reduction.h"
#include "cutlass/gemm/linear_scaling.h"

namespace cutlass {
namespace reduction {

/*
OutputTile defines the work load per thread block
Subtile defines the work load per thread block per iteration
OutputTile / Subtile = number of iterations within a kernel
ThreadShape defines the work load per thread
Subtile / ThreadShape = number of threads per thread block
*/
template <
  /// The scalar type for A
  typename ScalarA_,
  /// The scalar type for C
  typename ScalarC_,
  /// The scalar type for D
  typename ScalarD_,
  /// the scalar type for alpha,
  typename ScalarAlphaBeta_,
  /// The scalar type for accumulator
  typename ScalarAccum_,
  /// Reduction work load per batch
  int ReductionSize_ = 1,
  /// The output tile, work load per thread block,
  typename OutputTile_ = Shape<1, 1, 128>,
  /// The subtile
  typename SubTile_ = Shape<1, 1, 64>,
  /// Work load per thread, per subtile
  typename ThreadShape_ = Shape<1, 1, 2>,
  /// The index
  typename Index_ = int,
  /// The block swizzle to reorganize the grid.
  typename BlockSwizzle_ = DefaultBlockSwizzle,
  /// The input register vector size in kernel
  int maxInReg_ = 160,
  /// The output register vector size in kernel
  int maxOutReg_ = 64,
  /// The functor that will be executed at the end
  typename Functor_ = typename cutlass::gemm::LinearScaling<ScalarAlphaBeta_, typename cutlass::gemm::FragmentMultiplyAdd<ScalarAlphaBeta_, ScalarAccum_, (ThreadShape_::kW % 2 == 0)> >
>
struct BatchedReductionTraits {
  ///
  typedef BatchedReductionTraits<ScalarA_,
    ScalarC_,
    ScalarD_,
    ScalarAlphaBeta_,
    ScalarAccum_,
    ReductionSize_,
    OutputTile_,
    SubTile_,
    ThreadShape_,
    Index_,
    BlockSwizzle_,
    maxInReg_,
    maxOutReg_,
    Functor_> This_;
  /// The struct that consumes this Traits 
  typedef typename cutlass::reduction::BatchedReduction<This_> KernelClass;
  ///
  typedef OutputTile_ OutputTile;
  ///
  typedef SubTile_ SubTile;
  ///
  typedef ThreadShape_ ThreadShape;
  /// The input pointer type
  typedef ScalarA_ ScalarA;
  ///
  typedef ScalarC_ ScalarC;
  /// The output pointer type
  typedef ScalarD_ ScalarD;
  /// The alpha beta type
  typedef ScalarAlphaBeta_ ScalarAlphaBeta;
  /// The type for accumulation
  typedef ScalarAccum_ ScalarAccum;
  /// The index
  typedef Index_ Index;
  /// The thread block swizzle
  typedef BlockSwizzle_ BlockSwizzle;
  ///
  static const int ReductionSize = ReductionSize_;
  /// check if threadShape is multiple of 2. 
  static const bool ThreadShapeMultiple2 = (ThreadShape::kW % 2 == 0);
  ///
  typedef Functor_ Functor;
  /// Parameteres object constructable on the host
  /// The number of threads per thread block. can be deduced
  static int const kThreads = SubTile::kW / ThreadShape::kW;
  //
  static int const maxInReg = maxInReg_;
  //
  static int const maxOutReg = maxOutReg_;
  //
  static_assert(SubTile::kW % ThreadShape::kW == 0, "cannot evenly distribute work load among threads");
  //
  static_assert(kThreads % 32 == 0, "threads per threadblock is not multiple of 32");
  //
  static_assert(OutputTile::kW % SubTile::kW == 0, "cannot evenly distribute work load among iterations");
  //
  static_assert(ReductionSize * ThreadShape::kW <= maxInReg, "ReductionSize * ThreadShape::kW should not be bigger than maxInReg");
  //
  static_assert(ThreadShape::kW <= maxOutReg, "ThreadShape::kW should not be bigger than maxOutReg");

  struct Params {
    /// The dimension of output tensor 
    Coord<3> problem_size;
    /// The alpha
    ScalarAlphaBeta alpha;
    /// The beta
    ScalarAlphaBeta beta;
    /// stride between two element that will be sumed
    long long int reduction_stride;
    //
    ScalarA const *d_a;
    //
    Index lda;
    //
    ScalarC const *d_c;
    //
    Index ldc;
    //
    ScalarD *d_d;
    //
    Index ldd;
    /// The functor params.
    typename Functor::Params functorParams;
    /// Initialize the parameters for 2D output tensor
    CUTLASS_HOST_DEVICE int initialize(Index m_,
                                       Index n_,
                                       ScalarAlphaBeta alpha_,
                                       ScalarAlphaBeta beta_,
                                       long long int reduction_stride_,
                                       ScalarA const *d_a_,
                                       Index lda_,
                                       ScalarC const *d_c_,
                                       Index ldc_,
                                       ScalarD *d_d_,
                                       Index ldd_){
      problem_size = make_Coord(1, n_, m_);
      alpha = alpha_;
      beta = beta_;
      reduction_stride = reduction_stride_;
      d_a = d_a_;
      lda = lda_;
      d_c = d_c_;
      d_d = d_d_;
      ldc = ldc_;
      ldd = ldd_;

      functorParams.initialize(alpha_, beta_);

      return 0;
    }
  };

};
} // namespace reduction
} // namespace cutlass
