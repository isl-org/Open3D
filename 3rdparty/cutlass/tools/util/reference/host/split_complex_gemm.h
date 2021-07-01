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
    \brief Reference implementation for split-complex GEMM in device-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm_coord.h"
#include "cutlass/util/complex.h"

namespace cutlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a complex-valued GEMM whose operands are in the split-complex format.
template <
  typename TensorRefA,      /// concept: ZipTensorRef
  typename TensorRefB,      /// concept: ZipTensorRef
  typename TensorRefC,      /// concept: ZipTensorRef
  typename ScalarType,      /// real-valued type underlying complex scalars
  typename AccumulatorType  /// real-valued type underlying complex accumulators
>
void SplitComplexGemm(
  gemm::GemmCoord problem_size,
  platform::complex<ScalarType> alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  platform::complex<ScalarType> beta,
  TensorRefC tensor_c,
  platform::complex<AccumulatorType> initial_accum) {

  typedef typename TensorRefA::First::Storage AType;
  typedef typename TensorRefB::First::Storage BType;
  typedef typename TensorRefC::First::Storage CType;

  typedef platform::complex<AType> ComplexAType;
  typedef platform::complex<BType> ComplexBType;
  typedef platform::complex<CType> ComplexCType;
  typedef platform::complex<ScalarType> ComplexScalarType;
  typedef platform::complex<AccumulatorType> ComplexAccumulatorType;

  static_assert(
    TensorRefA::First::kRank == 2 && TensorRefA::Second::kRank == 2 &&
    TensorRefB::First::kRank == 2 && TensorRefB::Second::kRank == 2 &&
    TensorRefC::First::kRank == 2 && TensorRefC::Second::kRank == 2,
    "Tensors must be of rank 2");

  // Note: batch is ignored.
  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  // Blocking necessary to speedup reference implementation
  int const Mblock = 32;
  int const Nblock = 32;

  for (int row_block = 0; row_block < M; row_block += Mblock) {
    for (int col_block = 0; col_block < N; col_block += Nblock) {

      ComplexAccumulatorType accum[Mblock][Nblock];

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Mblock; i++) {
          accum[i][j] = initial_accum;
        }
      }

      for (int k_block = 0; k_block < K; ++k_block) {
        for (int j = 0; j < Nblock; j++) {
          for (int i = 0; i < Mblock; i++) {
            int row = row_block + i;
            int col = col_block + j;

            if (row < M && col < N) {

              ComplexAType a(
                tensor_a.first.at(MatrixCoord(row, k_block)),
                tensor_a.second.at(MatrixCoord(row, k_block))
              );

              ComplexBType b(
                tensor_b.first.at(MatrixCoord(k_block, col)),
                tensor_b.second.at(MatrixCoord(k_block, col))
              );

              accum[i][j] = detail::inner_product(a, b, accum[i][j]);
            }
          }
        }
      }

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Mblock; i++) {
          int row = row_block + i;
          int col = col_block + j;

          MatrixCoord coord = MatrixCoord(row, col);
          if (row < M && col < N) {

            ComplexScalarType product(
              detail::Cast<AccumulatorType, ScalarType>::apply(accum[i][j].real()),
              detail::Cast<AccumulatorType, ScalarType>::apply(accum[i][j].imag())
            );

            ComplexScalarType source(
              detail::Cast<CType, ScalarType>::apply(tensor_c.first.at(coord)),
              detail::Cast<CType, ScalarType>::apply(tensor_c.second.at(coord))
            );

            ComplexScalarType result = alpha * product + beta * source;

            tensor_c.first.at(coord) = detail::Cast<ScalarType, CType>::apply(result.real());
            tensor_c.second.at(coord) = detail::Cast<ScalarType, CType>::apply(result.imag());
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a complex-valued GEMM whose operands are in the split-complex format.
template <
  typename TensorRefA,      /// concept: ZipTensorRef
  typename TensorRefB,      /// concept: ZipTensorRef
  typename TensorRefC,      /// concept: ZipTensorRef
  typename ScalarType,      /// real-valued type underlying complex scalars
  typename AccumulatorType  /// real-valued type underlying complex accumulators
>
void SplitComplexGemm(
  gemm::GemmCoord problem_size,
  platform::complex<ScalarType> alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  platform::complex<ScalarType> beta,
  TensorRefC tensor_c) {

  return SplitComplexGemm(problem_size, alpha, tensor_a, tensor_b,beta, tensor_c, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Batched Split-Complex GEMM
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a complex-valued GEMM whose operands are in the split-complex format.
template <
  typename TensorRefCollectionA,      /// concept: Pair<TensorRefCollection, TensorRefCollection>
  typename TensorRefCollectionB,      /// concept: Pair<TensorRefCollection, TensorRefCollection>
  typename TensorRefCollectionC,      /// concept: Pair<TensorRefCollection, TensorRefCollection>
  typename ScalarType,                /// real-valued type underlying complex scalars
  typename AccumulatorType            /// real-valued type underlying complex accumulators
>
void BatchedSplitComplexGemm(
  gemm::GemmCoord problem_size,
  platform::complex<ScalarType> alpha,
  TensorRefCollectionA tensor_a,
  TensorRefCollectionB tensor_b,
  platform::complex<ScalarType> beta,
  TensorRefCollectionC tensor_c,
  platform::complex<AccumulatorType> initial_accum) {

  typename TensorRefCollectionA::ConstIterator tensor_a_real = tensor_a.first.begin();
  typename TensorRefCollectionA::ConstIterator tensor_a_imag = tensor_a.second.begin();

  typename TensorRefCollectionB::ConstIterator tensor_b_real = tensor_b.first.begin();
  typename TensorRefCollectionB::ConstIterator tensor_b_imag = tensor_b.second.begin();

  typename TensorRefCollectionC::ConstIterator tensor_c_real = tensor_c.first.begin();
  typename TensorRefCollectionC::ConstIterator tensor_c_imag = tensor_c.second.begin();

  for (int batch = 0; batch < problem_size.batch(); ++batch) {

    SplitComplexGemm(
      problem_size,
      alpha,
      make_ZipTensorRef(*tensor_a_real, *tensor_a_imag),
      make_ZipTensorRef(*tensor_b_real, *tensor_b_imag),
      beta,
      make_ZipTensorRef(*tensor_c_real, *tensor_c_imag),
      initial_accum);

    ++tensor_a_real;
    ++tensor_a_imag;
    ++tensor_b_real;
    ++tensor_b_imag;
    ++tensor_c_real;
    ++tensor_c_imag;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a complex-valued GEMM whose operands are in the split-complex format.
template <
  typename TensorRefCollectionA,      /// concept: pair<TensorRefCollection, TensorRefCollection>
  typename TensorRefCollectionB,      /// concept: pair<TensorRefCollection, TensorRefCollection>
  typename TensorRefCollectionC,      /// concept: pair<TensorRefCollection, TensorRefCollection>
  typename ScalarType,                /// real-valued type underlying complex scalars
  typename AccumulatorType            /// real-valued type underlying complex accumulators
>
void BatchedSplitComplexGemm(
  gemm::GemmCoord problem_size,
  platform::complex<ScalarType> alpha,
  TensorRefCollectionA tensor_a,
  TensorRefCollectionB tensor_b,
  platform::complex<ScalarType> beta,
  TensorRefCollectionC tensor_c) {

  BatchedSplitComplexGemm(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    platform::complex<ScalarType>(0, 0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
