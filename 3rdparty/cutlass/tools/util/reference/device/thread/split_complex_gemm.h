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

#include "tools/util/reference/detail/inner_product.h"

namespace cutlass {
namespace reference {
namespace device {
namespace thread {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Thread-level blocked general matrix product.
//
// Note, this is a reference implementation. Performance is not expected to approach peak.
//
template <
  typename TensorRefA,      /// concept: ZipTensorRef
  typename TensorRefB,      /// concept: ZipTensorRef
  typename TensorRefC,      /// concept: ZipTensorRef
  typename ScalarType,      /// real-valued type underlying complex scalars
  typename AccumulatorType, /// real-valued type underlying complex accumulators
  typename OutputTile       /// concept: Shape
>
struct SplitComplexGemm {

  typedef typename TensorRefA::First::Storage RealScalarA;
  typedef typename TensorRefB::First::Storage RealScalarB;
  typedef typename TensorRefC::First::Storage RealScalarC;

  typedef platform::complex<RealScalarA> ScalarA;
  typedef platform::complex<RealScalarB> ScalarB;
  typedef platform::complex<AccumulatorType> ComplexAccumulator;
  typedef platform::complex<ScalarType> ComplexScalar;

  //
  // Data members
  //

  /// Tile for A operand
  ScalarA A_tile[OutputTile::kW];

  /// Tile for B operand
  ScalarB B_tile[OutputTile::kH];

  /// Tile for Accumulator
  ComplexAccumulator accum[OutputTile::kH][OutputTile::kW];

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Gemm(ComplexAccumulator initial_accum = AccumulatorType(0)) {

    // Clear fetch registers
    for (int i = 0; i < OutputTile::kW; ++i) {
      A_tile[i] = ScalarA(0);
    }

    for (int j = 0; j < OutputTile::kW; ++j) {
      B_tile[j] = ScalarB(0);
    }

    // Clear accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < OutputTile::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTile::kW; ++i) {
        accum[j][i] = initial_accum;
      } 
    }
  }

  /// Computes a matrix product
  CUTLASS_HOST_DEVICE
  Gemm & multiply_add(
    gemm::GemmCoord problem_size,
    TensorRefA tensor_a,
    TensorRefB tensor_b,
    MatrixCoord output_coord = MatrixCoord()) {
    
    // Loop over the GEMM K dimension
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < problem_size.k(); ++k) {

      // Fetch a slice of the A matrix - zip into complex values
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < OutputTile::kW; ++i) {
        if (output_coord.row() + i < problem_size.m()) {
          MatrixCoord coord(output_coord.row() + i, k);
          A_tile[i].real() = tensor_a.first.at(coord);
          A_tile[i].imag() = tensor_a.second.at(coord);
        }
      }

      // Fetch a slice of the B matrix - zip into complex values
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < OutputTile::kH; ++j) {
        if (output_coord.column() + j < problem_size.n()) {
          MatrixCoord coord(k, output_coord.column() + j);
          B_tile[j].real() = tensor_b.first.at(coord);
          B_tile[j].imag() = tensor_b.second.at(coord);
        }
      }

      // Compute an accumulated matrix product on complex values
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < OutputTile::kH; ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < OutputTile::kW; ++i) {
          accum[j][i] = detail::inner_product(A_tile[i], B_tile[j], accum[j][i]);
        }
      }
    }

    return *this;
  }

  /// Performs linear scaling of matrix product and updates output tensor
  CUTLASS_HOST_DEVICE
  Gemm & epilogue(
    gemm::GemmCoord problem_size,
    ComplexScalar alpha,
    ComplexScalar beta,
    TensorRefC tensor_c,
    MatrixCoord output_coord = MatrixCoord()) {
    
    // Update the output tensor
    for (int j = 0; j < OutputTile::kH; ++j) {
      for (int i = 0; i < OutputTile::kW; ++i) {
        MatrixCoord coord = output_coord + MatrixCoord(i, j);
        if (coord < problem_size.mn()) {
          
          ComplexScalar source(
            tensor_c.first.at(coord),
            tensor_c.second.at(coord)
          );

          // Final calculation is performed in data type of scalars
          ComplexScalar result = alpha * ComplexScalar(accum[j][i].real(), accum[j][i].imag()) + beta * source;

          // Unzip and convert into output tensor data type
          tensor_c.first.at(coord) = detail::Cast<ScalarType, RealScalarC>::apply(result.real());
          tensor_c.second.at(coord) = detail::Cast<ScalarType, RealScalarC>::apply(result.imag());
        }
      }
    }

    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace device
} // namespace reference
} // namespace cutlass
