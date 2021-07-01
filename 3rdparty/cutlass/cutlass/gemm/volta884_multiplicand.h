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

#include "cutlass/gemm/mma_global_tile.h"
#include "cutlass/gemm/volta884_shared_tile.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines iterators for loading and storing multiplicands
template <
    /// Identifies multiplicand of GEMM (A or B)
    GemmOperand::Kind Operand,
    /// Specifies layout of data in source memory
    MatrixLayout::Kind Layout,
    /// Specifies threadblock tile shape
    typename Tile,
    /// Specifies warp tile shape
    typename WarpTile,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp tiles
    typename WarpDelta_>
struct Volta884Multiplicand;

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines iterators for loading and storing multiplicands for A.column_major
template <typename Tile_, typename WarpTile_, int WarpCount, typename WarpDelta_>
struct Volta884Multiplicand<GemmOperand::kA,
                            MatrixLayout::kColumnMajor,
                            Tile_,
                            WarpTile_,
                            WarpCount,
                            WarpDelta_> {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kA;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// Thread-block tile shape
  typedef Tile_ Tile;

  /// Warp-level matrix multiply-add shape
  typedef WarpTile_ WarpTile;

  /// Total number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp tiles
  typedef WarpDelta_ WarpDelta;

  //
  // Thread-block load iterator
  //
  typedef
      typename MMAThreadblockCongruousLoad<kOperand, Tile_, WarpCount, WarpDelta::kW>::Iterator
          LoadIterator;

  //
  // Thread-block store iterator
  //
  typedef Volta884ThreadblockMultiplicandStoreIterator<kOperand,
                                                       kLayout,
                                                       Tile_,
                                                       WarpCount,
                                                       WarpDelta::kW>
      StoreIterator;

  //
  // Warp-level load iterator
  //
  typedef Volta884WarpMultiplicandLoadIterator<kOperand,
                                               kLayout,
                                               Tile_,
                                               WarpTile_,
                                               WarpCount,
                                               WarpDelta>
      WarpLoadIterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines iterators for loading and storing multiplicands for B.row_major
template <typename Tile_, typename WarpTile_, int WarpCount, typename WarpDelta_>
struct Volta884Multiplicand<GemmOperand::kB,
                            MatrixLayout::kRowMajor,
                            Tile_,
                            WarpTile_,
                            WarpCount,
                            WarpDelta_> {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kB;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// Thread-block tile shape
  typedef Tile_ Tile;

  /// Warp-level matrix multiply-add shape
  typedef WarpTile_ WarpTile;

  /// Total number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp tiles
  typedef WarpDelta_ WarpDelta;

  //
  // Thread-block load iterator
  //
  typedef
      typename MMAThreadblockCongruousLoad<kOperand, Tile_, WarpCount, WarpDelta::kH>::Iterator
          LoadIterator;

  //
  // Thread-block store iterator
  //
  typedef Volta884ThreadblockMultiplicandStoreIterator<kOperand,
                                                       kLayout,
                                                       Tile_,
                                                       WarpCount,
                                                       WarpDelta::kH>
      StoreIterator;

  //
  // Warp-level load iterator
  //
  typedef Volta884WarpMultiplicandLoadIterator<kOperand,
                                               kLayout,
                                               Tile_,
                                               WarpTile_,
                                               WarpCount,
                                               WarpDelta>
      WarpLoadIterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines iterators for loading and storing multiplicands for A.row_major
template <typename Tile_, typename WarpTile_, int WarpCount, typename WarpDelta_>
struct Volta884Multiplicand<GemmOperand::kA,
                            MatrixLayout::kRowMajor,
                            Tile_,
                            WarpTile_,
                            WarpCount,
                            WarpDelta_> {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kA;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// Thread-block tile shape
  typedef Tile_ Tile;

  /// Warp-level matrix multiply-add shape
  typedef WarpTile_ WarpTile;

  /// Total number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp tiles
  typedef WarpDelta_ WarpDelta;

  //
  // Thread-block load iterator
  //
  typedef
      typename MMAThreadblockCrosswiseLoad<kOperand, Tile_, WarpCount, WarpDelta::kW>::Iterator
          LoadIterator;

  //
  // Thread-block store iterator
  //
  typedef Volta884ThreadblockMultiplicandStoreIterator<kOperand,
                                                       kLayout,
                                                       Tile_,
                                                       WarpCount,
                                                       WarpDelta::kW>
      StoreIterator;

  //
  // Warp-level load iterator
  //
  typedef Volta884WarpMultiplicandLoadIterator<kOperand,
                                               kLayout,
                                               Tile_,
                                               WarpTile_,
                                               WarpCount,
                                               WarpDelta>
      WarpLoadIterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines iterators for loading and storing multiplicands for B.row_major
template <typename Tile_, typename WarpTile_, int WarpCount, typename WarpDelta_>
struct Volta884Multiplicand<GemmOperand::kB,
                            MatrixLayout::kColumnMajor,
                            Tile_,
                            WarpTile_,
                            WarpCount,
                            WarpDelta_> {
  /// Identifies multiplicand of GEMM (A or B)
  static GemmOperand::Kind const kOperand = GemmOperand::kB;

  /// Specifies layout of data in source memory
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// Thread-block tile shape
  typedef Tile_ Tile;

  /// Warp-level matrix multiply-add shape
  typedef WarpTile_ WarpTile;

  /// Total number of participating warps
  static int const kWarpCount = WarpCount;

  /// Delta between warp tiles
  typedef WarpDelta_ WarpDelta;

  //
  // Thread-block load iterator
  //
  typedef
      typename MMAThreadblockCrosswiseLoad<kOperand, Tile_, WarpCount, WarpDelta::kH>::Iterator
          LoadIterator;

  //
  // Thread-block store iterator
  //
  typedef Volta884ThreadblockMultiplicandStoreIterator<kOperand,
                                                       kLayout,
                                                       Tile_,
                                                       WarpCount,
                                                       WarpDelta::kH>
      StoreIterator;

  //
  // Warp-level load iterator
  //
  typedef Volta884WarpMultiplicandLoadIterator<kOperand,
                                               kLayout,
                                               Tile_,
                                               WarpTile_,
                                               WarpCount,
                                               WarpDelta>
      WarpLoadIterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
