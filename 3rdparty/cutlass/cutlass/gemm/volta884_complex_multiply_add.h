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
    \brief Implements warp-level multiply-accumulate operations using Volta's mma.sync instruction
      for complex-valued data types.
*/

#pragma once

#include "cutlass/util/complex.h"
#include "cutlass/zip_fragment.h"
#include "cutlass/gemm/volta884_multiply_add.h"
#include "cutlass/zip_fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Shape of a warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_,
    /// Layout of multiplicand A
    MatrixLayout::Kind LayoutA,
    /// Indicates matrix transform on multiplicand A
    MatrixTransform::Kind TransformA,
    /// Data type of multiplicand A
    typename ScalarA_,
    /// Layout of multiplicand B
    MatrixLayout::Kind LayoutB,
    /// Indicates matrix transform on multiplicand B
    MatrixTransform::Kind TransformB,
    /// Data type of multiplicand B
    typename ScalarB_,
    /// Data type of accumulators
    typename ScalarC_,
    /// If true, A operand is conjugated
    bool ConjugateA = false,
    /// If true, B operand is conjugated
    bool ConjugateB = false,
    /// If true, infinite results are saturated to +-MAX_FLOAT
    bool SatFinite = false>
struct Volta884ComplexMultiplyAdd {
  //
  // Constant and type definitions
  //

  /// Shape of a warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;

  /// Shape of a warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ AccumulatorsPerWarp;

  /// Most of the Volta884 code assumes interleaved 32x32 tiles
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Shape of an individual warp-wide mma.sync instruction
  typedef Shape<4, 16, 16> InstructionShape;

  /// Shape of a warp-level matrix multiply operation
  typedef Shape<InstructionShape::kD, WarpGemmShape::kH, WarpGemmShape::kW> WarpTile;

  /// Verify WarpTile is a multiple of fundamental 32x32 interleaved tile
  static_assert(!(WarpTile::kH % InterleavedTileShape::kH) &&
                    !(WarpTile::kW % InterleavedTileShape::kW) && WarpTile::kD == 4,
                "WarpTile must be a multiple of InterleavedTileShape.");

  /// Layout of A multiplicand
  static MatrixLayout::Kind const kLayoutA = LayoutA;

    /// Indicates matrix transform on multiplicand B
  static MatrixTransform::Kind const kTransformA = TransformA;

  /// Layout of B multiplicand
  static MatrixLayout::Kind const kLayoutB = LayoutB;

    /// Indicates matrix transform on multiplicand B
  static MatrixTransform::Kind const kTransformB = TransformB;

  /// The type for A.
  typedef ScalarA_ ScalarA;
  /// The type for B.
  typedef ScalarB_ ScalarB;
  /// The type for C and D.
  typedef ScalarC_ ScalarC;

  /// If true, infinite results are saturated to +-MAX_FLOAT
  static bool const kSatFinite = SatFinite;

  /// Hard-coded comptue type supported on Volta
  static arch::ComputeType::Kind const kComputeType = arch::ComputeType::kDefault;

  /// Underlying matrix multiply-add operator
  typedef Volta884MultiplyAdd<WarpGemmShape,
                              kLayoutA,
                              ScalarA,
                              kLayoutB,
                              ScalarB,
                              ScalarC>
      RealMultiplyAdd;

  /// Fragment definition for A multiplicand
  typedef ZipFragment<typename RealMultiplyAdd::FragmentA, typename RealMultiplyAdd::FragmentA>
      FragmentA;

  /// Fragment definition for B multiplicand
  typedef ZipFragment<typename RealMultiplyAdd::FragmentB, typename RealMultiplyAdd::FragmentB>
      FragmentB;

  /// Fragment definition for accumulators
  typedef ZipFragment<typename RealMultiplyAdd::Accumulators,
                      typename RealMultiplyAdd::Accumulators>
      Accumulators;

  /// Number of mma.sync operations performed. See Volta884MultiplyAdd::Iterations for details.
  typedef typename RealMultiplyAdd::Iterations Iterations;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE Volta884ComplexMultiplyAdd() {}

  /// Multiply : d = a*b.
  CUTLASS_DEVICE void multiply_add(FragmentA const& A,
                                   FragmentB const& B,
                                   Accumulators const& C,
                                   Accumulators& D) {
    RealMultiplyAdd op;

    // complex-valued multiply-add
    op.multiply_add(A.first, B.first, C.first, D.first);
    op.multiply_add(A.first, B.second, C.second, D.second, kTransformB == MatrixTransform::kConjugate);
    op.multiply_add(A.second, B.first, C.second, D.second, kTransformA == MatrixTransform::kConjugate);
    op.multiply_add(A.second, B.second, C.first, D.first,
      !((kTransformA == MatrixTransform::kConjugate) ^ (kTransformB == MatrixTransform::kConjugate)));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Complex-valued epilogue
template <typename Accumulator, typename WarpDelta, typename Iterations>
struct Volta884ComplexNaiveEpilogue {
  /// Accumulator data type
  typedef Accumulator ScalarC;

  /// Output accumulator type
  typedef Accumulator ScalarD;

  /// BLAS Scalar type
  typedef Accumulator Scalar;

  /// Real-valued epilogue
  typedef Volta884NaiveEpilogue<Accumulator, WarpDelta, Iterations> RealEpilogue;

  /// Params object
  struct Params {
    /// Parameters for the real-valued part
    typename RealEpilogue::Params real;

    /// Parameters for the imaginary-valued part
    typename RealEpilogue::Params imag;

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE Params() {}

    /// Constructs from params object
    CUTLASS_HOST_DEVICE Params(typename RealEpilogue::Params const& _real,
                               typename RealEpilogue::Params const& _imag)
        : real(_real), imag(_imag) {}

    /// Construct from pointers
    CUTLASS_HOST_DEVICE Params(ScalarC* _real, int _ldr, ScalarC* _imag, int _ldi)
        : real(_real, _ldr), imag(_imag, _ldi) {}

    /// Construct from pointers
    CUTLASS_HOST_DEVICE Params(
      platform::complex<Scalar> const &alpha,
      platform::complex<Scalar> const &beta,
      ScalarC const *real_C,
      int real_ldc,
      ScalarC const *imag_C,
      int imag_ldc,
      ScalarD *real_D,
      int real_ldd,
      ScalarD *imag_D,
      int imag_ldd
    ):
      real(real_D, real_ldd, alpha.real(), beta.real()),
      imag(imag_D, imag_ldd, alpha.real(), beta.real()) { }

    /// Initializer method
    CUTLASS_HOST_DEVICE
    int initialize(
      platform::complex<Scalar> const &alpha,
      platform::complex<Scalar> const &beta,
      ScalarC const *real_C,
      int real_ldc,
      ScalarC const *imag_C,
      int imag_ldc,
      ScalarD *real_D,
      int real_ldd,
      ScalarD *imag_D,
      int imag_ldd
    ) {

      real = typename RealEpilogue::Params(real_D, real_ldd, alpha.real(), beta.real());
      imag = typename RealEpilogue::Params(imag_D, imag_ldd, alpha.real(), beta.real());

      return 0;
    }
  };

  /// Shared stoarge
  struct SharedStorage {};

  /// Accumulator fragment definition
  typedef ZipFragment<
    typename RealEpilogue::Accumulators,
    typename RealEpilogue::Accumulators> Accumulators;

  //
  // Data members
  //

  /// Epilogue for real part
  RealEpilogue real;

  /// Epilogue for imaginary part
  RealEpilogue imag;

  //
  // Methods
  //

  /// Constructs a complex-valued epilogue
  CUTLASS_DEVICE Volta884ComplexNaiveEpilogue(
      Params const& _params, Coord<3> const& _problem_size = make_Coord(1024, 1024, 1024))
      : real(_params.real, _problem_size), imag(_params.imag, _problem_size) {}

  /// Constructs a complex-valued epilogue
  CUTLASS_DEVICE Volta884ComplexNaiveEpilogue(ScalarC* _real,
                                              int _ldr,
                                              ScalarC* _imag,
                                              int _ldi,
                                              Coord<3> const& _problem_size = make_Coord(1024,
                                                                                         1024,
                                                                                         1024))
      : real(_real, _ldr, _problem_size), imag(_imag, _ldi, _problem_size) {}

  /// Constructs a complex-valued epilogue
  CUTLASS_DEVICE Volta884ComplexNaiveEpilogue(Params const& _params,
                                              SharedStorage& shared_storage,
                                              Coord<3> const& _problem_size = make_Coord(1024,
                                                                                         1024,
                                                                                         1024))
      : real(_params.real, _problem_size), imag(_params.imag, _problem_size) {}

  /// Sets accumulators to zero
  CUTLASS_DEVICE void clear(Accumulators& C) {
    C.first.clear();
    C.second.clear();
  }

  /// Naive load operation for debugging
  CUTLASS_DEVICE void load(Accumulators& C,
                           Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    real.load(C.first, threadblock_offset);
    imag.load(C.second, threadblock_offset);
  }

  /// Naive store operation for debugging
  CUTLASS_DEVICE void store(Accumulators const& C,
                            Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    real.store(C.first, threadblock_offset);
    imag.store(C.second, threadblock_offset);
  }

  /// CUTLASS Epilogue interface
  CUTLASS_DEVICE void epilogue(Accumulators const& C,
                               Coord<3> const& threadblock_offset = make_Coord(0, 0, 0),
                               int batch_id = 0) {
    real.store(C.first, threadblock_offset);
    imag.store(C.second, threadblock_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
