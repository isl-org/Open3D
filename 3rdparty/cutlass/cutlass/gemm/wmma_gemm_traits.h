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
    \brief Defies structural properties of GEMM targeting WMMA API in CUDA.
*/
#pragma once

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API

#include "cutlass/convert.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/gemm_epilogue.h"
#include "cutlass/gemm/gemm_epilogue_traits.h"
#include "cutlass/gemm/gemm_global_tile.h"
#include "cutlass/gemm/gemm_shared_tile.h"
#include "cutlass/gemm/gemm_traits.h"
#include "cutlass/gemm/wmma_gemm_epilogue_traits.h"
#include "cutlass/gemm/wmma_gemm_global_tile.h"
#include "cutlass/gemm/wmma_gemm_multiply_add.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

  template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The tile size for the GEMM KxNxM.
    typename OutputTile_,
    /// The input type.
    typename ScalarA_,
    /// The input type.
    typename ScalarB_,
    /// The output type.
    typename ScalarC_,
    /// The accumulator type.
    typename Accumulator_,
    /// Tile size for warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_,
    /// The shape of the WMMA instruction.
    typename InstructionShape_,
    /// The number of scalars per LDG for A.
    int kScalarsPerLdgA_,
    /// The number of scalars per LDG for B.
    int kScalarsPerLdgB_,
    /// The number of scalars per LDS for A.
    int KScalarsPerLdsA_,
    /// The number of scalars per LDS for B.
    int KscalarsPerLdsB_,
    /// The number of scalars per LDG for C and STG for D.
    int kScalarsPerLdgCAndStgD_,
    /// The number of scalars per STS for D.
    int kScalarsPerStsD_,
    /// The number of scalars per LDS for D.
    int kScalarsPerLdsD_
>
struct WmmaGemmConfig : public GemmConfig<
                            /// The scalar type for A.
                            ScalarA_,
                            /// The scalar type for B.
                            ScalarB_,
                            /// The scalar type for C.
                            ScalarC_,
                            /// The scalar type for D.
                            ScalarC_,
                            /// The tile size for the GEMM KxNxM.
                            OutputTile_,
                            /// The functor to do the math in the main loop.
                            WmmaGemmMultiplyAdd<kLayoutA_,
                                                ScalarA_,
                                                kLayoutB_,
                                                ScalarB_,
                                                MatrixLayout::kColumnMajor,
                                                Accumulator_,
                                                WarpGemmShape_,
                                                InstructionShape_>,
                            /// The number of scalars per LDG for A.
                            kScalarsPerLdgA_,
                            /// The number of scalars per STS for A.
                            kScalarsPerLdgA_,
                            /// The number of scalars per LDS for A.
                            KScalarsPerLdsA_,
                            /// The number of scalars per LDG for B.
                            kScalarsPerLdgB_,
                            /// The number of scalars per STS for B.
                            kScalarsPerLdgB_,
                            /// The number of scalars per LDS for B.
                            KscalarsPerLdsB_,
                            /// The number of scalars per LDG for C and STG for D.
                            kScalarsPerLdgCAndStgD_,
                            /// The number of scalars per STS for D.
                            kScalarsPerStsD_,
                            /// The number of scalars per LDS for D.
                            kScalarsPerLdsD_,
                            /// The number of stages in shared memory.
                            1,
                            /// If true, residue is computed in mainloop. If false, separate loops are instantiated.
                            false,
                            /// Is residue performed in prologue?
                            true,
                            /// If true, kernel is launched with CUDA launch bounds specified
                            false> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_,
          typename GemmConfig_,
          typename ScalarA_>
struct WmmaGemmTileTraitsHelperA {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename ScalarA_>
struct WmmaGemmTileTraitsHelperA<MatrixLayout::kColumnMajor, GemmConfig_, ScalarA_>
    : public GemmTileTraitsHelperA<MatrixLayout::kColumnMajor, GemmConfig_> {
  /// The base config.
  typedef GemmTileTraitsHelperA<MatrixLayout::kColumnMajor, GemmConfig_> Base;

  /// The skew.
  static int const kSkew = 16 / sizeof(typename Base::MultiplyAddScalar);
  /// The shared tile size.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kD,
                GemmConfig_::OutputTile::kW + kSkew>
      Tile;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kColumnMajor,
                     typename Base::MultiplyAddScalar,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      typename Base::MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename Base::GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kW * GemmConfig_::Warps::kW;
  /// The number of scalars loaded per iteration.
  static int const kScalarsPerIteration = Tile::kW * GemmConfig_::InstructionShape::kD;
  /// The traits class to build the iterator to load from shared memory for A.
  typedef WmmaGemmSharedLoadTileATraits<
      // The layout of the matrix.
      MatrixLayout::kColumnMajor,
      // The pointer.
      typename Base::MultiplyAddScalar,
      // The output tile size.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kW / kScalarsPerW>,
      // The stride between iterations.
      Shape<kScalarsPerIteration, 0, kScalarsPerW, 0>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename ScalarA_>
struct WmmaGemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_, ScalarA_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     MultiplyAddScalar,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kW,
                GemmConfig_::OutputTile::kD + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kW * GemmConfig_::Warps::kW;
  /// The traits class to build the iterator to load from shared memory for A.
  typedef WmmaGemmSharedLoadTileATraits<
      // The layout of the matrix.
      MatrixLayout::kRowMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kW * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kW / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with binary operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_, Vector<bin1_t, 32> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'bits'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kBitsPerScalar' to get the number in 'Scalar'.
  static int const kBitsPerScalar = sizeof(Scalar) * 8;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<bin1_t, 32>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD / kBitsPerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kBitsPerScalar),
            GemmConfig_::OutputTile::kD / kBitsPerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA / kBitsPerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kW,
                GemmConfig_::OutputTile::kD / kBitsPerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA / kBitsPerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kW * GemmConfig_::Warps::kW;
  /// The traits class to build the iterator to load from shared memory for A.
  typedef WmmaGemmSharedLoadTileATraits<
      // The layout of the matrix.
      MatrixLayout::kRowMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kW * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kW / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kBitsPerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with unsigned 4-bit integer operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_, Vector<uint4_t, 8> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'int4'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kInt4PerScalar' to get the number in 'Scalar'.
  static int const kInt4PerScalar = sizeof(Scalar) * 2;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<uint4_t, 8>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kInt4PerScalar),
            GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA / kInt4PerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kW,
                GemmConfig_::OutputTile::kD / kInt4PerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA / kInt4PerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kW * GemmConfig_::Warps::kW;
  /// The traits class to build the iterator to load from shared memory for A.
  typedef WmmaGemmSharedLoadTileATraits<
      // The layout of the matrix.
      MatrixLayout::kRowMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kW * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kW / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kInt4PerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with signed 4-bit integer operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_, Vector<int4_t, 8> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'int4'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kInt4PerScalar' to get the number in 'Scalar'.
  static int const kInt4PerScalar = sizeof(Scalar) * 2;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kA,
                     MatrixLayout::kRowMajor,
                     Vector<int4_t, 8>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kInt4PerScalar),
            GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA / kInt4PerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kW,
                GemmConfig_::OutputTile::kD / kInt4PerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA / kInt4PerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kW * GemmConfig_::Warps::kW;
  /// The traits class to build the iterator to load from shared memory for A.
  typedef WmmaGemmSharedLoadTileATraits<
      // The layout of the matrix.
      MatrixLayout::kRowMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kW * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kW / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kInt4PerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_,
          typename GemmConfig_,
          typename ScalarB_>
struct WmmaGemmTileTraitsHelperB {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename ScalarB_>
struct WmmaGemmTileTraitsHelperB<MatrixLayout::kRowMajor, GemmConfig_, ScalarB_>
    : public GemmTileTraitsHelperB<MatrixLayout::kRowMajor, GemmConfig_> {
  /// The base config.
  typedef GemmTileTraitsHelperB<MatrixLayout::kRowMajor, GemmConfig_> Base;

  /// The skew.
  static int const kSkew = 16 / sizeof(typename Base::MultiplyAddScalar);
  /// The shared tile size.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kD,
                GemmConfig_::OutputTile::kH + kSkew>
      Tile;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kRowMajor,
                     typename Base::MultiplyAddScalar,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to store data to shared memory for B^T.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      typename Base::MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename Base::GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kH * GemmConfig_::Warps::kH;
  /// The number of scalars loaded per iteration.
  static int const kScalarsPerIteration = Tile::kW * GemmConfig_::InstructionShape::kD;
  /// The traits class to build the iterator to load from shared memory for B.
  typedef WmmaGemmSharedLoadTileBTraits<
      // The layout of the matrix.
      MatrixLayout::kRowMajor,
      // The pointer.
      typename Base::MultiplyAddScalar,
      // The output tile size.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kH,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kH / kScalarsPerW>,
      // The stride between iterations.
      Shape<kScalarsPerIteration, 0, kScalarsPerW, 0>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename ScalarB_>
struct WmmaGemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_, ScalarB_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     MultiplyAddScalar,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // A is row-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kH,
                GemmConfig_::OutputTile::kD + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kH * GemmConfig_::Warps::kH;
  /// The traits class to build the iterator to load from shared memory for B.
  typedef WmmaGemmSharedLoadTileBTraits<
      // The layout of the matrix.
      MatrixLayout::kColumnMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kH * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kH / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with binary operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_, Vector<bin1_t, 32> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'bits'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kBitsPerScalar' to get the number in 'Scalar'.
  static int const kBitsPerScalar = sizeof(Scalar) * 8;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<bin1_t, 32>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // A is row-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD / kBitsPerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kBitsPerScalar),
            GemmConfig_::OutputTile::kD / kBitsPerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB / kBitsPerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kH,
                GemmConfig_::OutputTile::kD / kBitsPerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB / kBitsPerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kH * GemmConfig_::Warps::kH;
  /// The traits class to build the iterator to load from shared memory for B.
  typedef WmmaGemmSharedLoadTileBTraits<
      // The layout of the matrix.
      MatrixLayout::kColumnMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kH * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kH / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kBitsPerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with unsigned 4-bit integer operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_, Vector<uint4_t, 8> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'int4'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kInt4PerScalar' to get the number in 'Scalar'.
  static int const kInt4PerScalar = sizeof(Scalar) * 2;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<uint4_t, 8>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // A is row-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kInt4PerScalar),
            GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB / kInt4PerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kH,
                GemmConfig_::OutputTile::kD / kInt4PerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB / kInt4PerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kH * GemmConfig_::Warps::kH;
  /// The traits class to build the iterator to load from shared memory for B.
  typedef WmmaGemmSharedLoadTileBTraits<
      // The layout of the matrix.
      MatrixLayout::kColumnMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kH * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kH / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kInt4PerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_USE_SUBBYTE_WMMA
/// Specialization for WMMA GEMM with signed 4-bit integer operands
template <typename GemmConfig_>
struct WmmaGemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_, Vector<int4_t, 8> > {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// GemmConfig_::OutputTile::kD is in number of 'int4'. TileTraits expects number of 'Scalar'.
  /// Divide by 'kInt4PerScalar' to get the number in 'Scalar'.
  static int const kInt4PerScalar = sizeof(Scalar) * 2;

  /// WMMA matrix
  typedef WmmaMatrix<GemmOperand::kB,
                     MatrixLayout::kColumnMajor,
                     Vector<int4_t, 8>,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // A is row-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1,
            GemmConfig_::kThreads / (GemmConfig_::OutputTile::kD / kInt4PerScalar),
            GemmConfig_::OutputTile::kD / kInt4PerScalar>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB / kInt4PerScalar>
      GlobalTileTraits;

  /// The skew.
  static int const kSkew = 16 / sizeof(MultiplyAddScalar);
  /// The tile.
  typedef Shape<GemmConfig_::kStages,
                GemmConfig_::OutputTile::kH,
                GemmConfig_::OutputTile::kD / kInt4PerScalar + kSkew>
      Tile;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Tile,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB / kInt4PerScalar>
      SharedStoreTileTraits;

  /// The number of elements loaded in one LDG.
  static int const kScalarsPerW = GemmConfig_::InstructionShape::kH * GemmConfig_::Warps::kH;
  /// The traits class to build the iterator to load from shared memory for B.
  typedef WmmaGemmSharedLoadTileBTraits<
      // The layout of the matrix.
      MatrixLayout::kColumnMajor,
      // The pointer.
      MultiplyAddScalar,
      // The tile in shared memory.
      Tile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The strides between warps.
      GemmConfig_::InstructionShape::kH * Tile::kW,
      // The number of iterations to load the data.
      Shape<1, 1, GemmConfig_::OutputTile::kH / kScalarsPerW>,
      // The stride between iterations.
      Shape<GemmConfig_::InstructionShape::kD / kInt4PerScalar, 0, kScalarsPerW * Tile::kW>,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedLoadTileTraits;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The output tile.
    typename OutputTile_,
    /// The input type.
    typename ScalarA_,
    /// The input type.
    typename ScalarB_,
    /// The output type.
    typename ScalarC_,
    /// The accumulator type.
    typename Accumulator_,
    /// The functor to do the math in the epilogue.
    typename EpilogueFunctor_,
    /// Tile size for warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_,
    /// The shape of the WMMA instruction.
    typename InstructionShape_,
    /// The number of halfs loaded in one LDG for A.
    int kScalarsPerLdgA_,
    /// The number of halfs loaded in one LDG for B.
    int kScalarsPerLdgB_,
    /// The number of scalars per LDS for A.
    int KScalarsPerLdsA_,
    /// The number of scalars per LDS for B.
    int KscalarsPerLdsB_,
    /// The number of scalars per LDG for C and STG for D.
    int kScalarsPerLdgCAndStgD_,
    /// The number of scalars per STS for D.
    int kScalarsPerStsD_,
    /// The number of scalars per LDS for D.
    int kScalarsPerLdsD_,
    /// The index.
    typename Index_>
struct WmmaGemmTraitsHelper {
  /// The WMMA GEMM config.
  typedef WmmaGemmConfig<kLayoutA_,
                         kLayoutB_,
                         OutputTile_,
                         ScalarA_,
                         ScalarB_,
                         ScalarC_,
                         Accumulator_,
                         WarpGemmShape_,
                         InstructionShape_,
                         kScalarsPerLdgA_,
                         kScalarsPerLdgB_,
                         KScalarsPerLdsA_,
                         KscalarsPerLdsB_,
                         kScalarsPerLdgCAndStgD_,
                         kScalarsPerStsD_,
                         kScalarsPerLdsD_
                       >
      GemmConfig;

  /// The GEMM config for A.
  typedef WmmaGemmTileTraitsHelperA<kLayoutA_, GemmConfig, ScalarA_> GemmTileTraitsHelperA;
  /// The GEMM config for B.
  typedef WmmaGemmTileTraitsHelperB<kLayoutB_, GemmConfig, ScalarB_> GemmTileTraitsHelperB;

  /// The iterator to load A from global memory.
  typedef GemmGlobalIteratorAb<typename GemmTileTraitsHelperA::GlobalTileTraits, Index_>
      GlobalLoadIteratorA;
  /// The default transformer for A.
  typedef Copy<typename GlobalLoadIteratorA::Fragment> GlobalTransformerA;
  /// The iterator to store A to shared memory.
  typedef TileStoreIterator<typename GemmTileTraitsHelperA::SharedStoreTileTraits,
                            typename GemmTileTraitsHelperA::SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared>
      SharedStoreIteratorA;
  /// The stream to load A from global memory to shared memory.
  typedef GlobalLoadStream<GemmOperand::kA,
                              GlobalLoadIteratorA,
                              SharedStoreIteratorA,
                              GlobalTransformerA>
      GlobalLoadStreamA;

  /// The iterator to load B from global memory.
  typedef GemmGlobalIteratorAb<typename GemmTileTraitsHelperB::GlobalTileTraits, Index_>
      GlobalLoadIteratorB;
  // The default transformer for B.
  typedef Copy<typename GlobalLoadIteratorB::Fragment> GlobalTransformerB;
  /// The iterator to store B to shared memory.
  typedef TileStoreIterator<typename GemmTileTraitsHelperB::SharedStoreTileTraits,
                            typename GemmTileTraitsHelperB::SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared>
      SharedStoreIteratorB;
  /// The stream to load B from global memory to shared memory.
  typedef GlobalLoadStream<GemmOperand::kB,
                              GlobalLoadIteratorB,
                              SharedStoreIteratorB,
                              GlobalTransformerB>
      GlobalLoadStreamB;

  /// The iterator to load A from shared memory.
  typedef TileLoadIterator<typename GemmTileTraitsHelperA::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperA::SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared,
                           Index_,
                           typename GemmTileTraitsHelperA::WmmaMatrix,
                           FragmentElementType::kWmmaMatrix>
      SharedLoadIteratorA;
  /// The stream to load A from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorA> SharedLoadStreamA;
  /// The iterator to load B from shared memory.
  typedef TileLoadIterator<typename GemmTileTraitsHelperB::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperB::SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared,
                           Index_,
                           typename GemmTileTraitsHelperB::WmmaMatrix,
                           FragmentElementType::kWmmaMatrix>
      SharedLoadIteratorB;
  /// The stream to load B from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorB> SharedLoadStreamB;

  /// The functor to do the multiply-add in the main loop.
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;
  /// The object to clear accumulators.
  typedef ClearAccumulators<typename MultiplyAdd::ScalarC> ClearAccumulators;

  /// The helper to create the epilogue traits.
  typedef WmmaGemmEpilogueTraitsHelper<GemmConfig, Accumulator_, EpilogueFunctor_, Index_> EpilogueTraitsHelper;
  /// The traits class for the epilogue.
  typedef SimplifiedGemmEpilogueTraits<GemmConfig, EpilogueFunctor_, Index_, EpilogueTraitsHelper>
      GemmEpilogueTraits;
  /// The epilogue.
  typedef GemmEpilogue<GemmEpilogueTraits> Epilogue;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename OutputTile_, typename DefaultShape_ = Shape<64, 32, 64> >
struct WmmaGemmAccumulatorsPerWarp {
  typedef typename ShapeMin<OutputTile_, DefaultShape_>::Shape Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The tile size for the GEMM KxNxM.
    typename OutputTile_ = Shape<64, 128, 128>,
    /// The input type.
    typename ScalarA_ = half,
    /// The input type.
    typename ScalarB_ = half,
    /// The output type.
    typename ScalarC_ = float,
    /// The functor to do the math in the epilogue.
    typename EpilogueFunctor_ = LinearScaling<ScalarC_>,
    /// The accumulator type.
    typename Accumulator_ = ScalarC_,
    /// Tile size for warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_ = typename WmmaGemmAccumulatorsPerWarp<OutputTile_>::Shape,
    /// The shape of the WMMA instruction.
    typename InstructionShape_ = Shape<16, 16, 16>,
    /// The number of scalars per LDG for A.
    int kScalarsPerLdgA_ = 8,
    /// The number of scalars per LDG for B.
    int kScalarsPerLdgB_ = 8,
    /// The number of scalars per LDS for A.
    int KScalarsPerLdsA_ = 8,
    /// The number of scalars per LDS for B.
    int KscalarsPerLdsB_ = 8,
    /// The number of scalars per LDG for C and STG for D.
    int kScalarsPerLdgCAndStgD_ = 16 / sizeof(ScalarC_),
    /// The number of scalars per STS for D.
    int kScalarsPerStsD_ = 16 / sizeof(Accumulator_),
    /// The number of scalars per LDS for D.
    int kScalarsPerLdsD_ = 16 / sizeof(Accumulator_),
    /// The index.
    typename Index_ = int,
    /// The helper class.
    typename Helper_ = WmmaGemmTraitsHelper<kLayoutA_,
                                            kLayoutB_,
                                            OutputTile_,
                                            ScalarA_,
                                            ScalarB_,
                                            ScalarC_,
                                            Accumulator_,
                                            EpilogueFunctor_,
                                            WarpGemmShape_,
                                            InstructionShape_,
                                            kScalarsPerLdgA_,
                                            kScalarsPerLdgB_,
                                            KScalarsPerLdsA_,
                                            KscalarsPerLdsB_,
                                            kScalarsPerLdgCAndStgD_,
                                            kScalarsPerStsD_,
                                            kScalarsPerLdsD_,
                                            Index_> >
struct WmmaGemmTraits : public GemmTraits<
                            // The config.
                            typename Helper_::GemmConfig,
                            // The stream to load A from global memory to shared memory.
                            typename Helper_::GlobalLoadStreamA,
                            // The stream to load B from global memory to shared memory.
                            typename Helper_::GlobalLoadStreamB,
                            // The stream to load A from shared memory.
                            typename Helper_::SharedLoadStreamA,
                            // The stream to load B from shared memory.
                            typename Helper_::SharedLoadStreamB,
                            // The epilogue.
                            typename Helper_::Epilogue,
                            // The block swizzle to reorganize the grid.
                            IdentityBlockSwizzle,
                            // The index.
                            Index_,
                            // The tool used to clear accumulators.
                            typename Helper_::ClearAccumulators> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

#endif  // defined CUTLASS_USE_WMMA_API
