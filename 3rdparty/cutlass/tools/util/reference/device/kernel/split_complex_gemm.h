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
    \brief Reference implementation for GEMM in host-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm_coord.h"
#include "cutlass/util/complex.h"

#include "tools/util/reference/device/thread/split_complex_gemm.h"

namespace cutlass {
namespace reference {
namespace device {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename TensorRefA,      /// concept: ZipTensorRef
  typename TensorRefB,      /// concept: ZipTensorRef
  typename TensorRefC,      /// concept: ZipTensorRef
  typename ScalarType,      /// real-valued type underlying complex scalars
  typename AccumulatorType, /// real-valued type underlying complex accumulators
  typename OutputTile       /// concept: Shape
>
__global__ void SplitComplexGemm(
  gemm::GemmCoord problem_size,
  platform::complex<ScalarType> alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  platform::complex<ScalarType> beta,
  TensorRefC tensor_c,
  platform::complex<AccumulatorType> initial_accum) {

  // Map each thread to a unique tile of the output matrix
  MatrixCoord output_coord(
    (threadIdx.x + blockIdx.x * blockDim.x) * OutputTile::kW,
    (threadIdx.y + blockIdx.y * blockDim.y) * OutputTile::kH
  );

  // Compute the general matrix product
  thread::Gemm<
    TensorRefA,
    TensorRefB,
    TensorRefC,
    ScalarType,
    AccumulatorType,
    OutputTile
  > gemm(initial_accum);

  gemm.multiply_add(
    problem_size, 
    tensor_a, 
    tensor_b, 
    output_coord);

  gemm.epilogue(problem_size, alpha, beta, tensor_c, output_coord);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace device
} // namespace reference
} // namespace cutlass
