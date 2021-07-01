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
#include "cutlass/util/complex.h"

#include "tools/util/reference/device/kernel/gemm.h"

namespace cutlass {
namespace reference {
namespace device {

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
  platform::complex<ScalarType> initial_accum) {

  static_assert(
    TensorRefA::First::kRank == 2 && TensorRefA::Second::kRank == 2 &&
    TensorRefB::First::kRank == 2 && TensorRefB::Second::kRank == 2 &&
    TensorRefC::First::kRank == 2 && TensorRefC::Second::kRank == 2,
    "Tensors must be of rank 2");

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
  kernel::SplitComplexGemm<
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

} // namespace device
} // namespace reference
} // namespace cutlass
