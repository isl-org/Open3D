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
    \brief Defines structural properties for GEMM targeting Volta's mma.sync instruction
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_iterator.h"
#include "cutlass/util/platform.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Iterators used to load multiplicands from global memory specialized for Volta884 access patterns
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator for loading data for congruous access patterns
template <GemmOperand::Kind Operand, typename Tile_, int WarpCount, int WarpDelta>
struct MMAThreadblockCongruousLoad {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = Operand;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout =
      (Operand == GemmOperand::kA ? MatrixLayout::kColumnMajor : MatrixLayout::kRowMajor);

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Projects the threadblock tile
  typedef typename gemm::GemmMultiplicandTraits<Tile_, Operand, kLayout>::Shape OperandShape;

  /// Reshapes the threadblock tile by access size
  typedef typename ReshapeTile<OperandShape, kAccessSize>::Tile VectorizedShape;

  /// Shape of tile
  typedef Shape<1, 4, 8> WarpStoreCoverage;

  /// Shape of tile loaded by each warp per load operation
  typedef Shape<1, 4, 8> WarpLoadShape;

  //
  // Load iterator
  //

  ///
  typedef Shape<1, WarpLoadShape::kH * kWarpCount, WarpLoadShape::kW> Delta;

  typedef Shape<0, 0, 0, 0> ImmediateOffsetStrides;

  /// Rakes warps along contiguous dimensions and strip-mines strided
  /// dimension.
  typedef Shape<1,
                VectorizedShape::kH / WarpStoreCoverage::kH / WarpCount,
                VectorizedShape::kW / WarpStoreCoverage::kW,
                1>
      Iterations;

  /// Functor computing starting offset for each thread
  struct ThreadOffset {
    __device__ Coord<4> operator()() const {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      int lane_k = lane_id / WarpLoadShape::kW;
      int lane_outer = lane_id % WarpLoadShape::kW;

      Coord<4> offset = make_Coord(0, warp_id * WarpLoadShape::kH + lane_k, lane_outer, 0);

      return offset;
    }
  };

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> LoadTileTraits;

  /// Load iterator
  typedef TileLoadIterator<LoadTileTraits, half, IteratorAdvance::kH> Iterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator for loading data for congruous access patterns
template <GemmOperand::Kind Operand, typename Tile_, int WarpCount, int WarpDelta>
struct MMAThreadblockCrosswiseLoad {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = Operand;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout =
      (Operand == GemmOperand::kA ? MatrixLayout::kRowMajor : MatrixLayout::kColumnMajor);

  /// Shape of thread-block multiplicand
  typedef Tile_ Tile;

  /// Number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp accumulator tiles along the outer dimension
  static int const kWarpDelta = WarpDelta;

  /// This implementation is specialized for 128b loads
  static int const kAccessSize = 8;

  /// Projects the threadblock tile
  typedef typename gemm::GemmMultiplicandTraits<Tile_, kOperand, kLayout>::Shape OperandShape;

  /// Reshapes the threadblock tile by access size
  typedef typename ReshapeTile<OperandShape, kAccessSize>::Tile VectorizedShape;

  /// Shape of tile
  typedef Shape<1, 8, 4> WarpStoreCoverage;

  /// Shape of tile loaded by each warp per load operation
  typedef Shape<1, 8, 4> WarpLoadShape;

  //
  // Load iterator
  //

  ///
  typedef Shape<1, WarpLoadShape::kH, WarpLoadShape::kW> Delta;

  typedef Shape<0, 0, 0, 0> ImmediateOffsetStrides;

  /// Rakes warps along contiguous dimensions and strip-mines strided
  /// dimension.
  typedef Shape<1,
                VectorizedShape::kH / WarpStoreCoverage::kH / WarpCount,
                VectorizedShape::kW / WarpStoreCoverage::kW,
                1>
      Iterations;

  /// Functor computing starting offset for each thread
  struct ThreadOffset {
    __device__ Coord<4> operator()() const {

      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);

      int lane_k = lane_id % WarpLoadShape::kW;
      int lane_outer = lane_id / WarpLoadShape::kW;

      Coord<4> offset =
          make_Coord(0, warp_id * Iterations::kH * WarpLoadShape::kH + lane_outer, lane_k, 0);

      return offset;
    }
  };

  /// Source tile traits
  typedef TileTraits<VectorizedShape, Delta, Iterations, ThreadOffset, kAccessSize> LoadTileTraits;

  /// Load iterator
  typedef TileLoadIterator<LoadTileTraits, half, IteratorAdvance::kW> Iterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // gemm 
}  // namespace cutlass
