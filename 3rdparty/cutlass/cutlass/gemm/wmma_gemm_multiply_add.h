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
    \brief Implements warp-level matrix multiply-accumulate operation using CUDA WMMA API.
*/
#pragma once

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API
#include "cutlass/fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MatrixLayout::Kind kLayoutA_,
          typename ScalarA_,
          MatrixLayout::Kind kLayoutB_,
          typename ScalarB_,
          MatrixLayout::Kind kLayoutC_,
          typename ScalarC_,
          typename WarpGemmShape_,
          typename InstructionShape_>
struct WmmaGemmMultiplyAdd {
  /// The shape of the instruction.
  typedef InstructionShape_ InstructionShape;
  /// The number of threads per warp. That's a dummy configuration.
  typedef Shape<1, InstructionShape_::kH, InstructionShape_::kW> ThreadsPerWarp;
  /// Dimensions of the warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;
  /// Aliased for compatibility. Will be removed in CUTLASS v2.0
  typedef WarpGemmShape_ AccumulatorsPerWarp;
  /// The type for A.
  typedef ScalarA_ ScalarA;
  /// The type for B.
  typedef ScalarB_ ScalarB;
  /// The type for C and D.
  typedef ScalarC_ ScalarC;
  /// The number of iterations.
  typedef typename ShapeDiv<AccumulatorsPerWarp, InstructionShape>::Shape Iterations;

  /// The element for A.
  typedef WmmaMatrix<GemmOperand::kA, kLayoutA_, ScalarA, InstructionShape> ElementA;
  /// The fragment for A.
  typedef Fragment<ElementA, Iterations::kW> FragmentA;

  /// The element for B.
  typedef WmmaMatrix<GemmOperand::kB, kLayoutB_, ScalarB, InstructionShape> ElementB;
  /// The fragment for B.
  typedef Fragment<ElementB, Iterations::kH> FragmentB;

  /// The element for C.
  typedef WmmaMatrix<GemmOperand::kC, kLayoutC_, ScalarC, InstructionShape> ElementC;
  /// The fragment for C.
  typedef Fragment<ElementC, Iterations::kH * Iterations::kW> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE WmmaGemmMultiplyAdd() {}

  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Iterations::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Iterations::kW; ++i) {
        // The input elements.
        ElementA const& elt_a = a[i];
        ElementB const& elt_b = b[j];
        ElementC const& elt_c = c[j * Iterations::kW + i];

        // The output element.
        ElementC& elt_d = d[j * Iterations::kW + i];

        // The wmma instruction.
        nvcuda::wmma::mma_sync(elt_d, elt_a, elt_b, elt_c);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with binary operands
template<typename WarpGemmShape_>
struct WmmaGemmMultiplyAdd <MatrixLayout::kRowMajor,
                            Vector<bin1_t, 32>,
                            MatrixLayout::kColumnMajor,
                            Vector<bin1_t, 32>,
                            MatrixLayout::kColumnMajor,
                            int,
                            WarpGemmShape_,
                            Shape<128, 8, 8> >{
  /// The shape of the instruction.
  typedef Shape<128, 8, 8> InstructionShape;
  /// The number of threads per warp. That's a dummy configuration.
  typedef Shape<1, 4, 8> ThreadsPerWarp;
  /// Dimensions of the warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;
  /// Aliased for compatibility. Will be removed in CUTLASS v2.0
  typedef WarpGemmShape_ AccumulatorsPerWarp;
  /// The type for A.
  typedef Vector<bin1_t, 32> ScalarA;
  /// The type for B.
  typedef Vector<bin1_t, 32> ScalarB;
  /// The type for C and D.
  typedef int ScalarC;
  /// The number of iterations.
  typedef typename ShapeDiv<AccumulatorsPerWarp, InstructionShape>::Shape Iterations;

  /// The element for A.
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<bin1_t, 32>,
                     InstructionShape> ElementA;
  /// The fragment for A.
  typedef Fragment<ElementA, Iterations::kW> FragmentA;

  /// The element for B.
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<bin1_t, 32>,
                     InstructionShape> ElementB;
  /// The fragment for B.
  typedef Fragment<ElementB, Iterations::kH> FragmentB;

  /// The element for C.
  typedef WmmaMatrix<GemmOperand::kC,
                     MatrixLayout::kColumnMajor,
                     int,
                     InstructionShape> ElementC;
  /// The fragment for C.
  typedef Fragment<ElementC, Iterations::kH * Iterations::kW> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE WmmaGemmMultiplyAdd() {}

  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Iterations::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Iterations::kW; ++i) {
        // The input elements.
        ElementA const& elt_a = a[i];
        ElementB const& elt_b = b[j];
        ElementC const& elt_c = c[j * Iterations::kW + i];

        // The output element.
        ElementC& elt_d = d[j * Iterations::kW + i];

        // The wmma instruction.
        nvcuda::wmma::bmma_sync(elt_d,
                                elt_a,
                                elt_b,
                                elt_c,
                                nvcuda::wmma::experimental::bmmaBitOpXOR,
                                nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
      }
    }
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with signed 4-bit integer operands
template<typename WarpGemmShape_>
struct WmmaGemmMultiplyAdd <MatrixLayout::kRowMajor,
                            Vector<int4_t, 8>,
                            MatrixLayout::kColumnMajor,
                            Vector<int4_t, 8>,
                            MatrixLayout::kColumnMajor,
                            int,
                            WarpGemmShape_,
                            Shape<32, 8, 8> >{
  /// The shape of the instruction.
  typedef Shape<32, 8, 8> InstructionShape;
  /// The number of threads per warp. That's a dummy configuration.
  typedef Shape<1, 4, 8> ThreadsPerWarp;
  /// Dimensions of the warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;
  /// Aliased for compatibility. Will be removed in CUTLASS v2.0
  typedef WarpGemmShape_ AccumulatorsPerWarp;
  /// The type for A.
  typedef Vector<int4_t, 8> ScalarA;
  /// The type for B.
  typedef Vector<int4_t, 8> ScalarB;
  /// The type for C and D.
  typedef int ScalarC;
  /// The number of iterations.
  typedef typename ShapeDiv<AccumulatorsPerWarp, InstructionShape>::Shape Iterations;

  /// The element for A.
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<int4_t, 8>,
                     InstructionShape> ElementA;
  /// The fragment for A.
  typedef Fragment<ElementA, Iterations::kW> FragmentA;

  /// The element for B.
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<int4_t, 8>,
                     InstructionShape> ElementB;
  /// The fragment for B.
  typedef Fragment<ElementB, Iterations::kH> FragmentB;

  /// The element for C.
  typedef WmmaMatrix<GemmOperand::kC,
                     MatrixLayout::kColumnMajor,
                     int,
                     InstructionShape> ElementC;
  /// The fragment for C.
  typedef Fragment<ElementC, Iterations::kH * Iterations::kW> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE WmmaGemmMultiplyAdd() {}

  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Iterations::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Iterations::kW; ++i) {
        // The input elements.
        ElementA const& elt_a = a[i];
        ElementB const& elt_b = b[j];
        ElementC const& elt_c = c[j * Iterations::kW + i];

        // The output element.
        ElementC& elt_d = d[j * Iterations::kW + i];

        // The wmma instruction.
        nvcuda::wmma::mma_sync(elt_d, elt_a, elt_b, elt_c);
      }
    }
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with unsigned 4-bit integer operands
template<typename WarpGemmShape_>
struct WmmaGemmMultiplyAdd <MatrixLayout::kRowMajor,
                            Vector<uint4_t, 8>,
                            MatrixLayout::kColumnMajor,
                            Vector<uint4_t, 8>,
                            MatrixLayout::kColumnMajor,
                            int,
                            WarpGemmShape_,
                            Shape<32, 8, 8> >{
  /// The shape of the instruction.
  typedef Shape<32, 8, 8> InstructionShape;
  /// The number of threads per warp. That's a dummy configuration.
  typedef Shape<1, 4, 8> ThreadsPerWarp;
  /// Dimensions of the warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;
  /// Aliased for compatibility. Will be removed in CUTLASS v2.0
  typedef WarpGemmShape_ AccumulatorsPerWarp;
  /// The type for A.
  typedef Vector<uint4_t, 8> ScalarA;
  /// The type for B.
  typedef Vector<uint4_t, 8> ScalarB;
  /// The type for C and D.
  typedef int ScalarC;
  /// The number of iterations.
  typedef typename ShapeDiv<AccumulatorsPerWarp, InstructionShape>::Shape Iterations;

  /// The element for A.
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<uint4_t, 8>,
                     InstructionShape> ElementA;
  /// The fragment for A.
  typedef Fragment<ElementA, Iterations::kW> FragmentA;

  /// The element for B.
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<uint4_t, 8>,
                     InstructionShape> ElementB;
  /// The fragment for B.
  typedef Fragment<ElementB, Iterations::kH> FragmentB;

  /// The element for C.
  typedef WmmaMatrix<GemmOperand::kC,
                     MatrixLayout::kColumnMajor,
                     int,
                     InstructionShape> ElementC;
  /// The fragment for C.
  typedef Fragment<ElementC, Iterations::kH * Iterations::kW> Accumulators;

  /// Ctor.
  CUTLASS_DEVICE WmmaGemmMultiplyAdd() {}

  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& a,
                                   FragmentB const& b,
                                   Accumulators const& c,
                                   Accumulators& d) {

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Iterations::kH; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Iterations::kW; ++i) {
        // The input elements.
        ElementA const& elt_a = a[i];
        ElementB const& elt_b = b[j];
        ElementC const& elt_c = c[j * Iterations::kW + i];

        // The output element.
        ElementC& elt_d = d[j * Iterations::kW + i];

        // The wmma instruction.
        nvcuda::wmma::mma_sync(elt_d, elt_a, elt_b, elt_c);
      }
    }
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

#endif  // defined CUTLASS_USE_WMMA_API
