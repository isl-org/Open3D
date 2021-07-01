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
    \brief Defies structural properties of half-precision GEMM computation.
*/
#pragma once

#include "cutlass/convert.h"
#include "cutlass/reshape_tile.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/gemm_epilogue.h"
#include "cutlass/gemm/gemm_epilogue_traits.h"
#include "cutlass/gemm/gemm_global_tile.h"
#include "cutlass/gemm/gemm_shared_tile.h"
#include "cutlass/gemm/gemm_traits.h"
#include "cutlass/gemm/hgemm_global_tile.h"
#include "cutlass/gemm/hgemm_multiply_add.h"
#include "cutlass/layout/thread/transform.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The tile size for the GEMM KxNxM.
    typename OutputTile_,
    /// Tile size for thread-level GEMM (K-by-N-by-M)
    typename ThreadGemmShape_,
    /// The number of scalars per LDG for A.
    int kScalarsPerLdgA_ = 2,
    /// The number of scalars per LDG for B.
    int kScalarsPerLdgB_ = 2>
struct HgemmConfig : public GemmConfig<
                         /// The scalar type for A.
                         half,
                         /// The scalar type for B.
                         half,
                         /// The scalar type for C.
                         half,
                         /// The scalar type for D.
                         half,
                         /// The tile size for the GEMM KxNxM.
                         OutputTile_,
                         /// The functor to do the math in the main loop.
                         ThreadMultiplyAdd<ThreadGemmShape_, Shape<1, 4, 8>, half, half, half>,
                         /// The number of scalars per LDG for A.
                         kScalarsPerLdgA_,
                         /// The number of scalars per STS for A.
                         kScalarsPerLdgA_,
                         /// The number of scalars per LDS for A.
                         8,
                         /// The number of scalars per LDG for B.
                         kScalarsPerLdgB_,
                         /// The number of scalars per STS for B.
                         kScalarsPerLdgB_,
                         /// The number of scalars per LDS for B.
                         8,
                         /// The number of scalars per LDG for C and STG for D.
                         2,
                         /// The number of scalars per STS for D.
                         8,
                         /// The number of scalars per LDS for D.
                         2,
                         /// The number of stages in shared memory.
                         2,
                         /// kResidueSeparate
                         false,
                         /// kResidueInPrologue
                         true,
                         /// kLaunchBounds
                         false
                         > {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_, typename Iterator_>
struct HgemmTransformerA {};

template <typename Iterator_>
struct HgemmTransformerA<MatrixLayout::kColumnMajor, Iterator_> {
  typedef Convert<typename Iterator_::Fragment, typename Iterator_::Fragment> Transformer;
};

template <typename Iterator_>
struct HgemmTransformerA<MatrixLayout::kRowMajor, Iterator_> {
  typedef typename Iterator_::FragmentShape FragmentShape;
  typedef cutlass::layout::thread::Transform<FragmentShape, 2, half, cutlass::MatrixLayout::RowMajor, half, cutlass::MatrixLayout::ColumnMajor > Transformer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_, typename Iterator_>
struct HgemmTransformerB {};

template <typename Iterator_>
struct HgemmTransformerB<MatrixLayout::kRowMajor, Iterator_> {
  typedef Convert<typename Iterator_::Fragment, typename Iterator_::Fragment> Transformer;
};

template <typename Iterator_>
struct HgemmTransformerB<MatrixLayout::kColumnMajor, Iterator_> {
  typedef typename Iterator_::FragmentShape FragmentShape;
  typedef cutlass::layout::thread::Transform<FragmentShape, 2, half, cutlass::MatrixLayout::RowMajor, half, cutlass::MatrixLayout::ColumnMajor > Transformer;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_, typename GemmConfig_>
struct HgemmTileTraitsHelperA : public GemmTileTraitsHelperA<kLayout_, GemmConfig_> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct HgemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_>
    : public GemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_> {
  /// The base config.
  typedef GemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_> Base;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef HgemmCrosswiseGlobalTileTraits<
      GemmOperand::kA,
      // The layout.
      MatrixLayout::kRowMajor,
      // The pointer.
      half const,
      // The tile has size MxK in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as (threads / K ) x K (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc)
      GemmConfig_::kScalarsPerLdgA>
      GlobalTileTraits;

  static int const kSkewA = 128 / sizeof(half) / GlobalTileTraits::Threads::kW / 2;

  /// The traits class to build the iterator to store data to shared memory for A^T.
  typedef GemmSharedStoreWithSkewTileAbTraits <
      // The pointer.
      half,
      // The tile has size KxM in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kW * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as warps x 32(the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      2,
      // The skew to avoid bank conflicts added in the tile W dimension.
      kSkewA<GemmConfig_::kScalarsPerLdsA ? GemmConfig_::kScalarsPerLdsA : kSkewA>
          SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for A^T.
  typedef GemmSharedLoadTileATraits<
      // The pointer.
      half const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      8,
      // The skew.
      SharedStoreTileTraits::kSkew>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind kLayout_, typename GemmConfig_>
struct HgemmTileTraitsHelperB : public GemmTileTraitsHelperB<kLayout_, GemmConfig_> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct HgemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_>
    : public GemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_> {
  /// The base config.
  typedef GemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_> Base;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef HgemmCrosswiseGlobalTileTraits<
      GemmOperand::kB,
      // The layout.
      MatrixLayout::kColumnMajor,
      // The pointer.
      half const,
      // The tile has size KxN in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc)
      GemmConfig_::kScalarsPerLdgB>
      GlobalTileTraits;

  static int const kSkewB = 128 / sizeof(half) / GlobalTileTraits::Threads::kW / 2;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreWithSkewTileAbTraits <
      // The pointer.
      half,
      // The tile has size KxN in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kH * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      2,
      // The skew to avoid bank conflicts added in the tile W dimension.
      kSkewB<GemmConfig_::kScalarsPerLdsB ? GemmConfig_::kScalarsPerLdsB : kSkewB>
          SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for B^N.
  typedef GemmSharedLoadTileBTraits<
      // The pointer.
      half const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      8,
      // The skew.
      SharedStoreTileTraits::kSkew>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The output tile.
    typename OutputTile_,
    /// The functor to do the math in the epilogue.
    typename EpilogueFunctor_,
    /// Tile size for thread-level GEMM (K-by-N-by-M)
    typename ThreadGemmShape_,
    /// The number of halfs loaded in one LDG for A.
    int kScalarsPerLdgA_ = 2,
    /// The number of halfs loaded in one LDG for B.
    int kScalarsPerLdgB_ = 2,
    /// The index.
    typename Index_ = int>
struct HgemmTraitsHelper {
  /// The HGEMM config.
  typedef HgemmConfig<OutputTile_, ThreadGemmShape_, kScalarsPerLdgA_, kScalarsPerLdgB_> GemmConfig;
  /// The GEMM config for A.
  typedef HgemmTileTraitsHelperA<kLayoutA_, GemmConfig> GemmTileTraitsHelperA;
  /// The GEMM config for B.
  typedef HgemmTileTraitsHelperB<kLayoutB_, GemmConfig> GemmTileTraitsHelperB;

  /// The iterator to load A from global memory.
  typedef GemmGlobalIteratorAb<typename GemmTileTraitsHelperA::GlobalTileTraits, Index_>
      GlobalLoadIteratorA;
  /// The default transformer for A.
  typedef typename HgemmTransformerA<GemmTileTraitsHelperA::kLayout,
                                     GlobalLoadIteratorA>::Transformer GlobalTransformerA;
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
  typedef typename HgemmTransformerB<GemmTileTraitsHelperB::kLayout,
                                     GlobalLoadIteratorB>::Transformer GlobalTransformerB;
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

  /// The iterator to load A from shared memory
  typedef TileLoadIterator<typename GemmTileTraitsHelperA::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperA::SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorA;
  /// The stream to load A from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorA> SharedLoadStreamA;
  /// The iterator to load B from shared memory.
  typedef TileLoadIterator<typename GemmTileTraitsHelperB::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperB::SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorB;
  /// The stream to load B from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorB> SharedLoadStreamB;

  /// The functor to do the multiply-add in the main loop.
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;
  /// The object to clear accumulators.
  typedef ClearAccumulators<typename MultiplyAdd::ScalarC> ClearAccumulators;

  /// The traits class for the epilogue.
  typedef SimplifiedGemmEpilogueTraits<GemmConfig, EpilogueFunctor_, Index_> GemmEpilogueTraits;
  /// The epilogue.
  typedef GemmEpilogue<GemmEpilogueTraits> Epilogue;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The output tile.
    typename OutputTile_ = Shape<8, 128, 128>,
    /// The functor to do the math in the epilogue.
    typename EpilogueFunctor_ = LinearScaling<half>,
    /// Tile size for warp-level GEMM (K-by-N-by-M)
    typename ThreadGemmShape_ = Shape<8, 8, 16>,
    /// The number of halfs loaded in one LDG for A.
    int kScalarsPerLdgA_ = 2,
    /// The number of halfs loaded in one LDG for B.
    int kScalarsPerLdgB_ = 2,
    /// The index.
    typename Index_ = int,
    /// The helper class.
    typename Helper_ = HgemmTraitsHelper<kLayoutA_,
                                         kLayoutB_,
                                         OutputTile_,
                                         EpilogueFunctor_,
                                         ThreadGemmShape_,
                                         kScalarsPerLdgA_,
                                         kScalarsPerLdgB_,
                                         Index_> >
struct HgemmTraits : public GemmTraits<
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
