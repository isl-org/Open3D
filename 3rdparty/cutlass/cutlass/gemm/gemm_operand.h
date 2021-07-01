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
    \brief Defines constant expressions for mapping GEMM problem size and strides onto pitch-linear
   memory.
*/
#pragma once

#include "cutlass/matrix_traits.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/util/platform.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to describe attributes of GEMM matrix operands
template <GemmOperand::Kind kOperand_, MatrixLayout::Kind kLayout_>
struct GemmOperandTraitsAb {
  static const bool Congruous =
      (kOperand_ == GemmOperand::kA ^ kLayout_ == MatrixLayout::kRowMajor);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmOperand::Kind kOperand_, typename Tile_>
struct GetExtent;

template <typename Tile_>
struct GetExtent<GemmOperand::kA, Tile_> {
  static const int kExtent = Tile_::kW;
};

template <typename Tile_>
struct GetExtent<GemmOperand::kB, Tile_> {
  static const int kExtent = Tile_::kH;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Determines the shape of a multiplicand tile in terms of strided (H) and contiguous (W)
/// dimensions
template <typename ThreadBlockTile_, GemmOperand::Kind Usage, MatrixLayout::Kind Layout>
struct GemmMultiplicandTraits {
  // Only defined for A or B
  static_assert(Usage == GemmOperand::kA || Usage == GemmOperand::kB,
                "MultiplicandTileShape defined only for A or B operands.");

  /// Shape of GEMM thread block tile (K, N, M)
  typedef ThreadBlockTile_ ThreadBlockTile;

  /// Identifies multiplicand
  static GemmOperand::Kind const kUsage = Usage;

  /// Layout of tile
  static MatrixLayout::Kind const kLayout = Layout;

  // True if K is the strided dimension
  static bool const kKstrided = (kUsage == GemmOperand::kA ^ kLayout == MatrixLayout::kRowMajor);

  /// Map the ThreadBlockShape onto (kH, kW) dimensions for A and B operand
  typedef typename platform::conditional<
      kKstrided,
      Shape<1, ThreadBlockTile::kD, GetExtent<Usage, ThreadBlockTile>::kExtent>,
      Shape<1, GetExtent<Usage, ThreadBlockTile>::kExtent, ThreadBlockTile::kD> >::type Shape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Project's a coordinate (K, N, M) onto inner and outer dimensions defined for each
/// operand.
template <GemmOperand::Kind operand, bool Kstrided = true>
struct ProjectOperand;

/// Project A operand - (0, K, M)
template <bool Kstrided>
struct ProjectOperand<GemmOperand::kA, Kstrided> {
  CUTLASS_HOST_DEVICE
  static Coord<3> project(Coord<3> const &coord) {
    if (Kstrided) {
      return make_Coord(0, coord[0], coord[2]);
    } else {
      return make_Coord(0, coord[2], coord[0]);
    }
  }
};

/// Project B operand - (0, K, N)
template <bool Kstrided>
struct ProjectOperand<GemmOperand::kB, Kstrided> {
  CUTLASS_HOST_DEVICE
  static Coord<3> project(Coord<3> const &coord) {
    if (Kstrided) {
      return make_Coord(0, coord[0], coord[1]);
    } else {
      return make_Coord(0, coord[1], coord[0]);
    }
  }
};

/// Project C operand - (0, N, M)
template <>
struct ProjectOperand<GemmOperand::kC, true> {
  CUTLASS_HOST_DEVICE
  static Coord<3> project(Coord<3> const &coord) { return make_Coord(0, coord[1], coord[2]); }
};

/// Project D operand - (0, N, M)
template <>
struct ProjectOperand<GemmOperand::kD, true> {
  CUTLASS_HOST_DEVICE
  static Coord<3> project(Coord<3> const &coord) { return make_Coord(0, coord[1], coord[2]); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
