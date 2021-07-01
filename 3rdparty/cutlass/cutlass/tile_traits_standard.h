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
    \brief Defines tile traits for several tile partitioning arrangements of threads expected to
      achieve efficient streaming performance.
*/
#pragma once

#include "cutlass/tile_iterator.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Basic thread offset function computed from a thread shape
template <typename ThreadShape>
struct TiledThreadOffset {
  /// Computes the logical coordinate from thread shape
  CUTLASS_HOST_DEVICE
  Coord<4> operator()() const {
    Coord<4> thread_offset;

    int index = threadIdx.x;

    thread_offset[3] = (index % ThreadShape::kC);
    index = (index / ThreadShape::kC);

    thread_offset[2] = (index % ThreadShape::kW);
    index = (index / ThreadShape::kW);

    thread_offset[1] = (index % ThreadShape::kH);
    index = (index / ThreadShape::kH);

    thread_offset[0] = index;

    return thread_offset;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tiling in which the number of threads is greater than the
/// contiguous dimension of the tile.
template <typename Tile_, int Threads>
struct TileTraitsStrideMajor {
  /// Shape of tile
  typedef Tile_ Tile;

  /// Number of participating threads
  static int const kThreads = Threads;

  // Static assertions
  static_assert(!(ShapeCount<Tile>::kDhw % kThreads),
                "Tiling undefined if elements not divisible by threads.");

  static_assert(Tile::kW <= kThreads,
                "This specialization assumes there are more threads than the contiguous dimension "
                "of the tile.");

  /// Shape of threads
  typedef Shape<1, kThreads / Tile::kW, Tile::kW, 1> ThreadShape;

  /// Delta along each dimension
  typedef Shape<1, ThreadShape::kH, 1, 1> Delta;

  /// Number of iterations
  typedef Shape<1, Tile::kH / ThreadShape::kH, 1, 1> Iterations;

  /// Computes the initial offset
  typedef TiledThreadOffset<ThreadShape> ThreadOffset;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tiling in which the number of threads is fewer than the tile size
/// in the contiguous dimension.
template <typename Tile_, int Threads>
struct TileTraitsContiguousMajor {
  /// Shape of tile
  typedef Tile_ Tile;

  /// Number of participating threads
  static int const kThreads = Threads;

  // Static assertions
  static_assert(Tile::kW >= kThreads,
                "This specialization assumes there are more threads than the contiguous dimension "
                "of the tile.");

  static_assert(!(ShapeCount<Tile>::kDhw % kThreads),
                "Tiling undefined if elements not divisible by threads.");

  static_assert(!(Tile::kW % kThreads),
                "The contiguous size of the tile must be divisible by the number of threads.");

  /// Thread shape
  typedef Shape<1, 1, kThreads> ThreadShape;

  /// Delta between each thread's access
  typedef Shape<1, 1, kThreads> Delta;

  /// Number of iterations
  typedef Shape<1, Tile::kH, Tile::kW / kThreads> Iterations;

  /// Computes the initial offset
  typedef TiledThreadOffset<ThreadShape> ThreadOffset;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tiling in which warps rake across the contiguous dimension
template <typename Tile_, int Threads, int AccessSize = 1>
struct TileTraitsWarpRake {
  /// Shape of tile
  typedef Tile_ Tile;

  /// Number of participating threads
  static int const kThreads = Threads;

  /// Hard-coded warp size
  static int const kWarpSize = 32;

  /// Number of participating warps
  static int const kWarpCount = kThreads / kWarpSize;

  // Static assertions
  static_assert(!(ShapeCount<Tile>::kDhw % kThreads),
                "Tiling undefined if elements not divisible by threads.");

  static_assert(!(kThreads % kWarpSize), "Number of threads must be divisible by the warp size.");

  static_assert(!(Tile::kW % kWarpSize), "Contiguous dimension must be divisible by the warp size");

  /// Warps strip-mined across strided dimension
  static int const kWarpsStrided = __NV_STD_MIN(kWarpCount, Tile::kH);

  /// Warps stripmined contiguous dimension
  static int const kWarpsContiguous = kWarpCount / kWarpsStrided;

  /// Arrangement of threads
  typedef Shape<1, kWarpsStrided, kWarpsContiguous * kWarpSize> ThreadShape;

  /// The same warp rakes along the contiguous dimension
  typedef Shape<1, kWarpsStrided, kWarpSize * AccessSize> Delta;

  /// Number of iterations
  typedef Shape<1, Tile::kH / Delta::kH, (Tile::kW / AccessSize) / ThreadShape::kW> Iterations;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    /// Basic thread offset function computed from a thread shape
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int tid = threadIdx.x;
      int warp = (tid / kWarpSize);
      int lane = (tid % kWarpSize);

      static int const kWarpSpanContiguous = kWarpSize * Iterations::kW;

      int warp_w = (warp % kWarpsContiguous);
      int warp_h = (warp / kWarpsContiguous);

      return make_Coord(0, warp_h, AccessSize * (lane + kWarpSpanContiguous * warp_w), 0);
    }
  };
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Chooses 'best' shape to enable warp raking along contiguous dimension if possible.
template <typename Tile_, int Threads>
struct TileTraitsStandard {
  /// Shape of tile
  typedef Tile_ Tile;

  /// Number of participating threads
  static int const kThreads = Threads;

  /// Hard-coded warp size
  static int const kWarpSize = 32;

  /// Number of participating warps
  static int const kWarpCount = kThreads / kWarpSize;

  /// By default, do not do scalar loads
  static int const kAccessSize = 1;

  // Static assertions
  static_assert(!(ShapeCount<Tile>::kDhw % kThreads),
                "Tiling undefined if elements not divisible by threads.");

  /// Choose the stride-major contiguous tiling if the contiguous dimension is
  /// smaller than the warp size. Otherwise, if it is divisible by the warp size,
  /// choose the warp rake arrangement.
  typedef typename platform::conditional <
      Tile::kW<kWarpSize,
               TileTraitsStrideMajor<Tile, Threads>,
               typename platform::conditional<!(Tile::kW % kWarpSize),
                                              TileTraitsWarpRake<Tile, Threads>,
                                              TileTraitsContiguousMajor<Tile, Threads> >::type>::
          type Traits;

  /// Delta between accesses
  typedef typename Traits::Delta Delta;

  /// Delta between each thread's access
  typedef Shape<0, 0, 0, 0> ImmediateOffsetStrides;

  /// Number of accesses
  typedef typename Traits::Iterations Iterations;

  /// Thread offset functor
  typedef typename Traits::ThreadOffset ThreadOffset;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
