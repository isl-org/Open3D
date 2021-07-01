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
    \brief Defines iterators for efficiently loading and storing tiles to and from shared memory.
*/
#pragma once

#include "cutlass/gemm/gemm_operand.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, typename Tile_, typename Threads_, int kScalarsPerSts_>
struct GemmSharedStoreTileAbTraits {
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The tile.
  typedef typename ReshapeTile<Tile_, kScalarsPerSts_>::Tile Tile;
  /// The threads.
  typedef Threads_ Threads;
  /// The strides to compute the base position of the thread.
  typedef Shape<0, ShapeCount<Tile>::kWc, Tile::kC, kScalarsPerSts_> ThreadsStrides;
  /// The skew.
  static int const kSkew = 0;
  /// The number of scalars per LDG/STG.
  static int const kAccessSize = kScalarsPerSts_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of iterations needed to load/store the tile.
  typedef Shape<1,
                Tile::kH / Threads::kH,
                Tile::kW / Threads::kW,
                Tile::kC / Threads::kC / kAccessSize>
      Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, Threads::kH * ShapeCount<Tile>::kWc, Threads::kW * kAccessSize> Delta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, Threads::kH * ShapeCount<Tile>::kWc, Threads::kW * kAccessSize>
      ImmediateOffsetStrides;

  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int offset = ComputeThreadOffsetFromStrides<Threads, ThreadsStrides>::get();
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_, typename Tile_, typename Threads_, int kScalarsPerSts_, int kSkew_>
struct GemmSharedStoreWithSkewTileAbTraits {
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The tile without skews.
  typedef typename ReshapeTile<Tile_, kScalarsPerSts_>::Tile TileWithoutSkew;
  /// The tile.
  typedef typename ReshapeTile<Shape<Tile_::kD, Tile_::kH, Tile_::kW + kSkew_>,
                               kScalarsPerSts_>::Tile Tile;
  /// The threads.
  typedef Threads_ Threads;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The number of scalars per STS.
  static int const kAccessSize = kScalarsPerSts_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of iterations needed to load/store the tile.
  typedef Shape<1, TileWithoutSkew::kH / Threads::kW, TileWithoutSkew::kW / Threads::kH> Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, ShapeCount<Tile>::kWc, Threads::kH * kAccessSize> Delta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, ShapeCount<Tile>::kWc, Threads::kH * kAccessSize> ImmediateOffsetStrides;

  struct ThreadOffset {
    CUTLASS_HOST_DEVICE Coord<4> operator()() const {
      int offset = ComputeThreadOffsetFromStrides<Threads, ThreadsStrides>::get();
      return make_Coord(0, 0, offset, 0);
    }
  };

 protected:
  /// The strides to compute the base position of the thread.
  typedef Shape<0, kScalarsPerSts_, ShapeCount<Tile>::kHwc / Threads::kW> ThreadsStrides;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          typename OutputTile_,
          typename Warps_,
          typename ThreadsPerWarp_,
          typename InstructionShape_,
          int kStages_,
          int kScalarsPerLds_,
          int kSkew_ = 0>
struct GemmSharedLoadTileATraits {
  static GemmOperand::Kind const kOperand = GemmOperand::kA;
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The tile without skew.
  typedef Shape<kStages_,
                OutputTile_::kD / InstructionShape_::kD,
                GetExtent<kOperand, OutputTile_>::kExtent * InstructionShape_::kD>
      TileWithoutSkew_;
  /// The tile with skew.
  typedef Shape<kStages_, TileWithoutSkew_::kH, TileWithoutSkew_::kW + kSkew_> TileWithSkew;
  /// The tile without skew after reshaping.
  typedef typename ReshapeTile<TileWithoutSkew_, kScalarsPerLds_>::Tile TileWithoutSkew;
  /// The tile.
  typedef typename ReshapeTile<TileWithSkew, kScalarsPerLds_>::Tile Tile;
  /// The number of warps.
  typedef Warps_ Warps;
  /// The threads in a warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of scalars per LDG/STG.
  // static int const kScalarsPerLds = kScalarsPerLds_;
  static int const kAccessSize = kScalarsPerLds_;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of warps.
  static int const kWarps = GetExtent<kOperand, Warps>::kExtent;
  /// The number of threads in one dimension of the warp.
  static int const kThreadsPerWarp = GetExtent<kOperand, ThreadsPerWarp>::kExtent;

  /// The number of iterations needed to load/store the tile.
  typedef Shape<1, 1, TileWithoutSkew::kW / kWarps / kThreadsPerWarp /* / kScalarsPerLds*/>
      Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<TileWithSkew::kW * Warps::kD, 0, kWarps * kThreadsPerWarp * kAccessSize, 0>
      ImmediateOffsetStrides;
  typedef Shape<TileWithSkew::kW * Warps::kD, 0, kWarps * kThreadsPerWarp * kAccessSize, 0> Delta;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE Coord<4> operator()() const {
      // Extract the warp.
      int const warp = threadIdx.x / kWarpSize;
      // Extract the slice.
      int const slice = warp / (Warps::kH * Warps::kW);
      // Compute the row offset for each warp.
      int const warp_row = warp % Warps::kW;
      // Compute the row offset for each thread.
      int const lane_row = (threadIdx.x & 0x0e) / 2;
      // The offset.
      int const offset =
          slice * Tile::kW * Tile::kC + (warp_row * ThreadsPerWarp::kW + lane_row) * kAccessSize;
      // Embed the offset in a 4D coordinate vector.
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          typename OutputTile_,
          typename Warps_,
          typename ThreadsPerWarp_,
          typename InstructionShape_,
          int kStages_,
          int kScalarsPerLds_,
          int kSkew_ = 0>
struct GemmSharedLoadTileBTraits {
  static GemmOperand::Kind const kOperand = GemmOperand::kB;
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The tile without skew.
  typedef Shape<kStages_,
                OutputTile_::kD / InstructionShape_::kD,
                GetExtent<kOperand, OutputTile_>::kExtent * InstructionShape_::kD>
      TileWithoutSkew_;
  /// The tile with skew.
  typedef Shape<kStages_, TileWithoutSkew_::kH, TileWithoutSkew_::kW + kSkew_> TileWithSkew;
  /// The tile without skew after reshaping.
  typedef typename ReshapeTile<TileWithoutSkew_, kScalarsPerLds_>::Tile TileWithoutSkew;
  /// The tile.
  typedef typename ReshapeTile<TileWithSkew, kScalarsPerLds_>::Tile Tile;
  /// The number of warps.
  typedef Warps_ Warps;
  /// The threads in a warp.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of scalars per LDG/STG.
  static int const kAccessSize = kScalarsPerLds_;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of warps.
  static int const kWarps = GetExtent<kOperand, Warps>::kExtent;
  /// The number of threads in one dimension of the warp.
  static int const kThreadsPerWarp = GetExtent<kOperand, ThreadsPerWarp>::kExtent;

  /// The number of iterations needed to load/store the tile.
  typedef Shape<1, 1, TileWithoutSkew::kW / kWarps / kThreadsPerWarp /* / kAccessSize*/> Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<TileWithSkew::kW * Warps::kD, 0, kWarps * kThreadsPerWarp * kAccessSize, 0>
      ImmediateOffsetStrides;
  typedef Shape<TileWithSkew::kW * Warps::kD, 0, kWarps * kThreadsPerWarp * kAccessSize, 0> Delta;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE Coord<4> operator()() const {
      // Extract the warp.
      int const warp = threadIdx.x / kWarpSize;
      // Extract the slice.
      int const slice = warp / (Warps::kH * Warps::kW);
      // The warp in the slice.
      int const warp_in_slice = warp % (Warps::kH * Warps::kW);
      // Compute the row offset for each warp.
      int const warp_col = warp_in_slice / Warps::kW;
      // Compute the row offset for each thread.
      int const lane_col = (threadIdx.x & 0x10) / 8 + (threadIdx.x & 0x01);
      // The offset.
      int const offset =
          slice * Tile::kW * Tile::kC + (warp_col * ThreadsPerWarp::kH + lane_col) * kAccessSize;
      // Embed the offset in a 4D coordinate.
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          typename OutputTile_,
          typename Warps_,
          typename ThreadsPerWarp_,
          int kScalarsPerSts_,
          int kSkew_ = 0>
struct GemmSharedStoreTileDTraits {
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The dimension of the output tile.
  typedef OutputTile_ OutputTile;
  /// The warps in the tile.
  typedef Warps_ Warps;
  /// The threads in the warps.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of scalars per LDG/STG.
  static int const kAccessSize = kScalarsPerSts_;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of scalars per thread.
  static int const kScalarsPerThread = OutputTile_::kW / Warps::kW / ThreadsPerWarp::kW;
  /// The number of threads.
  static int const kThreads = ShapeCount<Warps>::kCount * kWarpSize;
  /// The number of scalars per row. We build a tile with 2 rows (to avoid bank conflicts).
  static int const kScalarsPerRow = kThreads / 2 * kScalarsPerThread + kSkew;

  /// The tile.
  typedef Shape<1, 2, kScalarsPerRow / kAccessSize, kAccessSize> Tile;
  /// The number of iterations needed to store the tile.
  typedef Shape<1, 1, kScalarsPerThread / kAccessSize> Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Warps::kW * ThreadsPerWarp::kW * kAccessSize> Delta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<0, 0, Warps::kW * ThreadsPerWarp::kW * kAccessSize> ImmediateOffsetStrides;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE Coord<4> operator()() const {
      // The warp.
      int const warp = threadIdx.x / kWarpSize;

      // The position of the warp in the 2D tile.
      int const warp_row = warp % Warps::kW;
      int const warp_col = warp / Warps::kW;

      // We assume that the elements are distributed in a warps as 4 columns of 8 elements. The
      // columns are stored in threads col0=[0, 2, 4, 6, 8, 10, 12, 14], col1=[1, 3, 5, 7, .., 15],
      // col2=[16, 18, 20, ..., 30] and col3=[17, 19, ..., 31].
      int hi_halfwarp_offset = ((threadIdx.x >> 4) & 0x1) * OutputTile::kW;
      int lo_halfwarp_offset = ((threadIdx.x >> 1) & 0x7) + ThreadsPerWarp::kW * warp_row;

      // Odd threads go to the second half of shared memory.
      int const row = threadIdx.x & 0x01;
      int col = warp_col * (ThreadsPerWarp::kH / 2) * OutputTile::kW +
                lo_halfwarp_offset * kAccessSize + hi_halfwarp_offset;
      // Embed the offset in a 4D coords.
      return make_Coord(0, 0, row * kScalarsPerRow + col, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar_,
          typename OutputTile_,
          typename Warps_,
          typename ThreadsPerWarp_,
          int kTileH_,
          int kScalarsPerLds_,
          int kSkew_ = 0>
struct GemmSharedLoadTileDTraits {
  /// The scalar.
  typedef typename platform::remove_const<Scalar_>::type Scalar;
  /// The pointer.
  typedef Scalar_* Pointer;
  /// The dimension of the output tile.
  typedef OutputTile_ OutputTile;
  /// The warps in the tile.
  typedef Warps_ Warps;
  /// The threads in the warps.
  typedef ThreadsPerWarp_ ThreadsPerWarp;
  /// The number of scalars per LDG/STG.
  static int const kAccessSize = kScalarsPerLds_;
  /// The skew.
  static int const kSkew = kSkew_;
  /// The memory space.
  static MemorySpace::Kind const kMemorySpace = MemorySpace::kShared;

  /// The number of scalars per thread.
  static int const kScalarsPerThread = OutputTile_::kW / Warps::kW / ThreadsPerWarp::kW;
  /// The number of threads.
  static int const kThreads = ShapeCount<Warps>::kCount * kWarpSize;
  /// The number of scalars per row. We build a tile with 2 rows (to avoid bank conflicts).
  static int const kScalarsPerRow = kThreads / 2 * kScalarsPerThread + kSkew;

  /// The tile. We have 2 rows of scalars. We use those two rows to make sure we do not have bank
  /// conflicts in the epilogue.
  typedef Shape<1, 2, kScalarsPerRow / kAccessSize, kAccessSize> Tile;

  // Compute the number of iterations per warp in the Tile::kH dimension.
  static int const kIterationsInHPerWarp = kTileH_ / ShapeCount<Warps>::kCount;

  // As explained above, the shared memory tile is composed of 2 rows and each rows is made of
  // kScalarsPerRow. A warp is expected to read from the 1st row, then move to the 2nd row and go
  // back to the 1st row. To model that scheme we define the Iterations shape as Shape<X, 2, ...>.
  // However, in some cases, we have only 1 iteration per warp. In that case, we must define the
  // shape as Shape<1, 1, ...>. The following code does that except that we hijack the kH dimension
  // to keep the number of elements to reduce for split-K.
  static int const kIterationsH = kIterationsInHPerWarp == 1 ? 1 : 2;
  // As soon as we know kIterationsH, it is trivial to compute kIterationsD:
  static int const kIterationsD = kIterationsInHPerWarp / kIterationsH;

  // If we have split-K enabled, we have to jump over the elements from the "odd/even" column of
  // threads to grab the other elements.
  static int const kSplitK = OutputTile::kW * ThreadsPerWarp::kH / 2 * Warps::kH;

  /// The number of iterations needed to store the tile.
  typedef Shape<kIterationsD, kIterationsH, OutputTile::kW / kWarpSize / kAccessSize, Warps::kD>
      Iterations;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<OutputTile::kW, kScalarsPerRow, kWarpSize * kAccessSize, kSplitK>
      ImmediateOffsetStrides;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<OutputTile::kW, kScalarsPerRow, kWarpSize * kAccessSize, kSplitK> Delta;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE Coord<4> operator()() const {
      // Each warp works on a different column.
      int const h = threadIdx.x / kWarpSize;
      // Compute the row.
      int const w = (threadIdx.x & (kWarpSize - 1)) * kAccessSize;
      int offset = 0;
      if (Iterations::kH == 1) {
        int const row = h & 0x1;
        int const col = h / 2;
        offset = row * ShapeCount<Tile>::kWc + col * OutputTile::kW * Iterations::kD + w;
      } else {
        offset = h * OutputTile::kW * Iterations::kD + w;
      }
      return make_Coord(0, 0, offset, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
