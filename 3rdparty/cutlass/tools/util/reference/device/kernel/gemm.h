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

#include "tools/util/reference/device/thread/gemm.h"

namespace cutlass {
namespace reference {
namespace device {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename TensorRefA,
  typename TensorRefB,
  typename TensorRefC,
  typename ScalarType,
  typename AccumulatorType,
  typename OutputTile
>
__global__ void Gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefA tensor_a,
  TensorRefB tensor_b,
  ScalarType beta,
  TensorRefC tensor_c,
  AccumulatorType initial_accum) {

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

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename TensorRefCollectionA,
  typename TensorRefCollectionB,
  typename TensorRefCollectionC,
  typename ScalarType,
  typename AccumulatorType,
  typename OutputTile
>
__global__ void BatchedGemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRefCollectionA tensor_collection_a,
  TensorRefCollectionB tensor_collection_b,
  ScalarType beta,
  TensorRefCollectionC tensor_collection_c,
  AccumulatorType initial_accum) {

  // Obtain batch ID
  int batch_id = blockIdx.z;

  // Dereference based on batch_id
  typename TensorRefCollectionA::TensorRef tensor_a = tensor_collection_a.at(batch_id);
  typename TensorRefCollectionB::TensorRef tensor_b = tensor_collection_b.at(batch_id);
  typename TensorRefCollectionC::TensorRef tensor_c = tensor_collection_c.at(batch_id);

  // Map each thread to a unique tile of the output matrix
  MatrixCoord output_coord(
    (threadIdx.x + blockIdx.x * blockDim.x) * OutputTile::kW,
    (threadIdx.y + blockIdx.y * blockDim.y) * OutputTile::kH
  );

  // Compute the general matrix product
  thread::Gemm<
    typename TensorRefCollectionA::TensorRef,
    typename TensorRefCollectionB::TensorRef,
    typename TensorRefCollectionC::TensorRef,
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
