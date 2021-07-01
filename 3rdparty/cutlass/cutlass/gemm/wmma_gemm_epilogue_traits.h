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
    \brief Defines structural properties of WMMA GEMM's epilogue phase.
*/
#pragma once

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API

#include "cutlass/convert.h"
#include "cutlass/coord.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/linear_scaling.h"
#include "cutlass/gemm/wmma_gemm_global_tile.h"
#include "cutlass/gemm/wmma_gemm_shared_tile.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_iterator.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename Accumulator_, typename EpilogueFunctor_, typename Index_ = int>
struct WmmaGemmEpilogueTraitsHelper {
  /// The scalar.
  typedef typename EpilogueFunctor_::Scalar Scalar;
  /// The output tile.
  typedef typename GemmConfig_::OutputTile OutputTile;

  /// The number of WMMAs in the H dimension.
  static int const kWmmasPerH =
      GemmConfig_::AccumulatorsPerWarp::kH / GemmConfig_::InstructionShape::kH;
  /// The number of iterations in the epilogue. That's the number of "horizontal" WMMAs.
  typedef Shape<1, 1, kWmmasPerH> Iterations;
  // The iteration strides in the H/W dimension.
  typedef Shape<0, 0, 0> Delta;
  /// The functor to do the math in the epilogue.
  typedef EpilogueFunctor_ Functor;

  /// The traits class to build the iterator to store to shared memory for D.
  typedef WmmaGemmSharedStoreTileDTraits<
      // The output layout.
      MatrixLayout::kColumnMajor,
      // The pointer is float.
      typename Functor::Scalar,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The shape of the instruction.
      typename GemmConfig_::InstructionShape>
      SharedStoreTileTraits;

  typedef WmmaMatrix<GemmOperand::kC,
                     MatrixLayout::kColumnMajor,
                     Scalar,
                     typename GemmConfig_::InstructionShape>
      WmmaMatrix;

  /// The iterator to store D to shared memory.
  typedef TileStoreIterator<SharedStoreTileTraits,
                            typename SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared,
                            Index_,
                            WmmaMatrix,
                            FragmentElementType::kWmmaMatrix>
      SharedStoreIteratorD;

  /// The shared store transformer for D.
  typedef Copy<typename SharedStoreIteratorD::Fragment> SharedStoreTransformerD;

  /// The traits class to build the iterator to load from shared memory for D.
  typedef WmmaGemmSharedLoadTileDTraits<
      // The pointer.
      typename Functor::Scalar,
      // The tile size.
      typename SharedStoreIteratorD::Tile,
      // The number of threads.
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsD,
      // this parameter helps with swizzling when accum is fp32 and output is fp16
      int(sizeof(Accumulator_)) / int(sizeof(typename GemmConfig_::ScalarD)) 
      >
      SharedLoadTileTraits;

  /// The iterator to load D from shared memory.
  typedef TileLoadIterator<SharedLoadTileTraits,
                           typename SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorD;

  /// The stream to load D.
  typedef SharedLoadStream<SharedLoadIteratorD> SharedLoadStreamD;

  /// The traits class to build the iterator to load data from global memory for C^N.
  typedef WmmaGemmGlobalIteratorCdTraits<
      // The pointer is float const.
      typename GemmConfig_::ScalarC const,
      // The tile has size (N / Iterations)xM in GEMM's terminology.
      Shape<1,
            GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgC>
      GlobalLoadTileTraits;

  /// The iterator to load C.
  typedef WmmaGemmGlobalIteratorCd<GlobalLoadTileTraits, Index_> GlobalLoadIteratorC;
  /// The transformer for C.
  typedef Copy<typename GlobalLoadIteratorC::Fragment> GlobalTransformerC;

  /// The traits class to build the iterator to store data to global memory for D^N.
  typedef WmmaGemmGlobalIteratorCdTraits<
      // The pointer is float.
      typename GemmConfig_::ScalarD,
      // The tile has size (N / Iterations)xM in GEMM's terminology.
      Shape<1,
            GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerStgD>
      GlobalStoreTileTraits;

  /// The iterator to store D.
  typedef WmmaGemmGlobalIteratorCd<GlobalStoreTileTraits, Index_> GlobalStoreIteratorD;
  /// The transformer for D.
  typedef Copy<typename GlobalStoreIteratorD::Fragment> GlobalTransformerD;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

#endif  // defined CUTLASS_USE_WMMA_API
