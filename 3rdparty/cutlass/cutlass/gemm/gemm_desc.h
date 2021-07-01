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

#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/gemm_coord.h"

namespace cutlass {
namespace gemm {

/// GEMM problem description
template <
    /// Source accumulator matrix type
    typename AType_,
    /// Destination accumulator type
    typename BType_,
    /// Source accumulator matrix type
    typename CType_,
    /// Destination accumulator type
    typename DType_,
    /// Scalar type for alpha and beta
    typename SType_,
    /// Index type for dimensions and strides
    typename Index_ = int
> struct GemmDesc {
  //
  // Type definitions
  //

  /// Index type for dimensions and strides
  typedef Index_ Index;

  /// Source accumulator matrix type
  typedef AType_ AType;

  /// Tensor reference to A operand
  typedef TensorRef<AType const, 2> TensorRefA;

  /// Destination accumulator type
  typedef BType_ BType;

  /// Tensor reference to B operand
  typedef TensorRef<BType const, 2> TensorRefB;

  /// Source accumulator matrix type
  typedef CType_ CType;

  /// Tensor reference to C operand
  typedef TensorRef<CType const, 2> TensorRefC;

  /// Destination accumulator type
  typedef DType_ DType;

  /// Tensor reference to D operand
  typedef TensorRef<DType, 2> TensorRefD;

  /// Scalar type for alpha and beta
  typedef SType_ SType;

  //
  // Data members
  //

  /// The dimensions of the GEMM.
  GemmCoord problem_size;

  /// The alpha scaling values.
  SType alpha;

  /// The source matrix A.
  TensorRefA A;

  /// batch stride for A operand
  long long batch_stride_A;

  /// The source matrix B.
  TensorRefB B;

  /// batch stride for B operand
  long long batch_stride_B;

  /// The beta scaling values.
  SType beta;

  /// The source matrix C.
  TensorRefC C;

  /// batch stride for C operand
  long long batch_stride_C;

  /// The destination matrix D.
  TensorRefD D;

  /// batch stride for D operand
  long long batch_stride_D;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  GemmDesc(): problem_size(0, 0, 0, 1), alpha(1), beta(0) {}

  /// Constructor for basic GEMM with batch count = 1
  CUTLASS_HOST_DEVICE
  GemmDesc(Coord<3> _problem_size,
           SType _alpha,
           TensorRefA const &_A,
           TensorRefB const &_B,
           SType _beta,
           TensorRefC const &_C,
           TensorRefD const &_D
  ):
    problem_size(_problem_size[0], _problem_size[1], _problem_size[2], 1),
    alpha(_alpha),
    A(_A),
    batch_stride_A(0),
    B(_B),
    batch_stride_B(0),
    beta(_beta),
    C(_C),
    batch_stride_C(0),
    D(_D),
    batch_stride_D(0) {}

  /// Constructor for basic GEMM with batch count = 1
  CUTLASS_HOST_DEVICE
  GemmDesc(GemmCoord _problem_size,
           SType _alpha,
           TensorRefA const &_A,
           TensorRefB const &_B,
           SType _beta,
           TensorRefC const &_C,
           TensorRefD const &_D
  ):
    problem_size(_problem_size.k(), _problem_size.n(), _problem_size.m(), 1),
    alpha(_alpha),
    A(_A),
    batch_stride_A(0),
    B(_B),
    batch_stride_B(0),
    beta(_beta),
    C(_C),
    batch_stride_C(0),
    D(_D),
    batch_stride_D(0) {

    assert(_problem_size.batch() == 1);
  }

  /// Constructor for strided batch GEMM GEMM
  CUTLASS_HOST_DEVICE
  GemmDesc(GemmCoord _problem_size,
           SType _alpha,
           TensorRefA const &_A,
           long long _batch_stride_A,
           TensorRefB const &_B,
           long long _batch_stride_B,
           SType _beta,
           TensorRefC const &_C,
           long long _batch_stride_C,
           TensorRefD const &_D,
           long long _batch_stride_D
  ):
    problem_size(_problem_size),
    alpha(_alpha),
    A(_A),
    batch_stride_A(_batch_stride_A),
    B(_B),
    batch_stride_B(_batch_stride_B),
    beta(_beta),
    C(_C),
    batch_stride_C(_batch_stride_C),
    D(_D),
    batch_stride_D(_batch_stride_D) {}
};

}  // namespace gemm
}  // namespace cutlass
