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
    \brief Defines iterator traits for efficiently loading and storing fragment to and from shared
      memory, specialized for WMMA GEMM.
*/
#pragma once

#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API

#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/reshape_tile.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename Tile_,
          typename Warps_,
          int kWarpStride_,
          typename Iterations_,
          typename Delta_,
          typename WmmaShape_>
struct WmmaGemmSharedLoadTileATraits {
  /// The operand.
  static GemmOperand::Kind const kOperand = GemmOperand::kA;
  /// The layout.
  static MatrixLayout::Kind const kLayout = kLayout_;
  /// The scalar.
  typedef Scalar_ Scalar;
  /// The pointer.
  typedef Scalar const* Pointer;
  /// The access size
  static int const kAccessSize = 1;
  /// The tile with skew.
  typedef Tile_ Tile;
  /// The number of warps.
  typedef Warps_ Warps;
  /// The warps strides.
  static int const kWarpStride = kWarpStride_;
  /// The number of iterations.
  typedef Iterations_ Iterations;
  /// The strides between iterations.
  typedef Delta_ Delta;
  /// The strides between iterations.
  typedef Delta_ ImmediateOffsetStrides;
  /// The shape of the WMMA instruction.
  typedef WmmaShape_ WmmaShape;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;
  /// ThreadOffset
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      // The warp id.
      int const warp = threadIdx.x / kWarpSize;
      // The offset.
      int const offset = warp % Warps::kW * kWarpStride;
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename Tile_,
          typename Warps_,
          int kWarpStride_,
          typename Iterations_,
          typename Delta_,
          typename WmmaShape_>
struct WmmaGemmSharedLoadTileBTraits {
  /// The operand.
  static GemmOperand::Kind const kOperand = GemmOperand::kB;
  /// The layout.
  static MatrixLayout::Kind const kLayout = kLayout_;
  /// The scalar.
  typedef Scalar_ Scalar;
  /// The pointer.
  typedef Scalar const* Pointer;
  /// The access size
  static int const kAccessSize = 1;
  /// The tile with skew.
  typedef Tile_ Tile;
  /// The number of warps.
  typedef Warps_ Warps;
  /// The warps strides.
  static int const kWarpStride = kWarpStride_;
  /// The number of iterations.
  typedef Iterations_ Iterations;
  /// The strides between iterations.
  typedef Delta_ Delta;
  /// The strides between iterations.
  typedef Delta_ ImmediateOffsetStrides;
  /// The shape of the WMMA instruction.
  typedef WmmaShape_ WmmaShape;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;
  /// ThreadOffset
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      // The warp id.
      int const warp = threadIdx.x / kWarpSize;
      // The offset.
      int const offset = warp / Warps::kW * kWarpStride;
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename OutputTile_,
          typename Warps_,
          typename WmmaShape_,
          int kSkew_ = 0>
struct WmmaGemmSharedStoreTileDTraits {
  /// The operand.
  static GemmOperand::Kind const kOperand = GemmOperand::kC;
  /// The layout.
  static MatrixLayout::Kind const kLayout = kLayout_;
  /// The scalar.
  typedef Scalar_ Scalar;
  // The access size
  static int const kAccessSize = 1;
  /// The pointer.
  typedef Scalar* Pointer;
  /// The number of warps.
  typedef Warps_ Warps;
  /// The shape of the WMMA instruction.
  typedef WmmaShape_ WmmaShape;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;
  /// The tile with skew.
  typedef Shape<1, Warps_::kH * WmmaShape_::kH, OutputTile_::kW + kSkew_> Tile;
  /// The number of iterations needed to store the tile.
  typedef Shape<1, 1, OutputTile_::kW / Warps::kW / WmmaShape_::kW> Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Warps::kW * WmmaShape_::kW, 0> Delta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Warps::kW * WmmaShape_::kW, 0> ImmediateOffsetStrides;


  /// ThreadOffset
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      // The warp id.
      int const warp = threadIdx.x / kWarpSize;
      // The starting column.
      int const h = warp / Warps::kW * WmmaShape::kH;
      // The w.
      int const w = warp % Warps::kW * WmmaShape::kW;
      // The offset.
      int const offset = h * Tile::kW + w;
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, typename Tile_, typename Threads_, int kScalarsPerLds_, int kLdsPerAccess_ = 1>
struct WmmaGemmSharedLoadTileDTraits {
  /// The scalar.
  typedef Scalar_ Scalar;
  /// The pointer.
  typedef Scalar const* Pointer;
  /// The access size
  static int const kAccessSize = kScalarsPerLds_;
  /// The tile.
  typedef typename WmmaReshapeTile<Tile_, kScalarsPerLds_, kLdsPerAccess_>::Tile Tile;
  /// The threads.
  typedef typename ReshapeThreads<Tile, Threads_>::Threads Threads;
  /// The threads strides.
  typedef Shape<1, Tile::kW * Tile::kC, Tile::kC> ThreadsStrides;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, Threads::kH * ShapeCount<Tile>::kWc, Threads::kW * kScalarsPerLds_> Delta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, Threads::kH * ShapeCount<Tile>::kWc, Threads::kW * kScalarsPerLds_, kScalarsPerLds_>
      ImmediateOffsetStrides;
  /// The number of iterations needed to load/store the tile.
  typedef Shape<1, Tile::kH / Threads::kH, Tile::kW / Threads::kW, Tile::kC / kScalarsPerLds_>
      Iterations;


  /// ThreadOffset
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      // The offset.
      int const offset = ComputeThreadOffsetFromStrides<Threads, ThreadsStrides>::get();
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

#endif  // defined CUTLASS_USE_WMMA_API
