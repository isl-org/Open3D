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
    \brief Reference implementation for GEMM in device-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm_coord.h"

#include "tools/util/reference/device/kernel/gemm.h"

namespace cutlass {
namespace reference {
namespace device {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
///
/// Explicitly naming types needed by this template can be cumbersome, particularly for the
/// accumulator type, so a function argument 'initial_accum' is exposed. Passing
/// AccumulatorType(0) as the last function argument can be easier than naming all template
/// arguments explicitly.
template <
  typename TensorRefA,
  typename TensorRefB,
  typename TensorRefC,
  typename ScalarType,
  typename AccumulatorType
>
void Gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  ScalarType beta,
  TensorRefC tensor_c,
  AccumulatorType initial_accum) {

  typedef typename TensorRefA::Storage AType;
  typedef typename TensorRefB::Storage BType;
  typedef typename TensorRefC::Storage CType;

  static_assert(
    TensorRefA::kRank == 2 &&
    TensorRefB::kRank == 2 &&
    TensorRefC::kRank == 2, "Tensors must be of rank 2");

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  typedef Shape<1, 4, 4> OutputTile;

  dim3 block(16, 8);
  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kW - 1) / (block.x * OutputTile::kW),
    (problem_size.n() + block.y * OutputTile::kH - 1) / (block.y * OutputTile::kH)
  );

  // Launch a GEMM kernel
  kernel::Gemm<
    TensorRefA,
    TensorRefB,
    TensorRefC,
    ScalarType,
    AccumulatorType,
    OutputTile
  ><<< grid, block >>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    initial_accum
  );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
///
/// This assumes the accumulator type is the same type as the scalars.
template <
  typename TensorRefA,
  typename TensorRefB,
  typename TensorRefC,
  typename ScalarType
>
void Gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  ScalarType beta,
  TensorRefC tensor_c) {

  Gemm(problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Batched GEMM
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a batch of GEMMs over a set of matrices of common dimension.
//
// TensorRefCollection* is a type satisfying the TensorRefCollection concept.
//
template <
  typename TensorRefCollectionA,
  typename TensorRefCollectionB,
  typename TensorRefCollectionC,
  typename ScalarType,
  typename AccumulatorType
>
void BatchedGemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefCollectionA tensor_a,
  TensorRefCollectionB tensor_b,
  ScalarType beta,
  TensorRefCollectionC tensor_c,
  AccumulatorType initial_accum) {

  typedef typename TensorRefCollectionA::Storage AType;
  typedef typename TensorRefCollectionB::Storage BType;
  typedef typename TensorRefCollectionC::Storage CType;

  static_assert(
    TensorRefCollectionA::kRank == 2 &&
    TensorRefCollectionB::kRank == 2 &&
    TensorRefCollectionC::kRank == 2, "Tensors must be of rank 2");

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  typedef Shape<1, 4, 4> OutputTile;

  dim3 block(16, 8);
  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kW - 1) / (block.x * OutputTile::kW),
    (problem_size.n() + block.y * OutputTile::kH - 1) / (block.y * OutputTile::kH),
    problem_size.batch()
  );

  // Launch a GEMM kernel
  kernel::BatchedGemm<
    TensorRefCollectionA,
    TensorRefCollectionB,
    TensorRefCollectionC,
    ScalarType,
    AccumulatorType,
    OutputTile
  ><<< grid, block >>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    initial_accum
  );
}

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
//
// TensorRefCollection* is a type satisfying the TensorRefCollection concept.
//
template <
  typename TensorRefCollectionA,
  typename TensorRefCollectionB,
  typename TensorRefCollectionC,
  typename ScalarType,
  typename AccumulatorType
>
void BatchedGemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefCollectionA tensor_a,
  TensorRefCollectionB tensor_b,
  ScalarType beta,
  TensorRefCollectionC tensor_c) {

  BatchedGemm(problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
