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
// Warp-scoped shared memory load iterators
//
////////////////////////////////////////////////////////////////////////////////////////////////////

///! Iterator to store a thread-block scoped fragment to shared memory
template <
    /// Identifies multiplicand of GEMM (A or B)
    GemmOperand::Kind Operand,
    /// Specifies layout of data in source memory
    MatrixLayout::Kind Layout,
    /// Specifies threadblock tile shape
    typename Tile,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    int WarpDelta>
struct Volta884ThreadblockMultiplicandStoreIterator;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator to load a fragment for each warp-level tile
template <
    /// Identifies multiplicand of GEMM (A or B)
    GemmOperand::Kind Operand,
    /// Specifies layout of data in source memory
    MatrixLayout::Kind Layout,
    /// Specifies threadblock tile shape
    typename Tile,
    /// Specifies the warp tile shape
    typename WarpTile,
    /// Specifies the number of participating warps
    int WarpCount,
    /// Specifies the delta between warp accesses along the outer dimension
    typename WarpDelta>
struct Volta884WarpMultiplicandLoadIterator;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Fully-specialized implementations extracted in separate headers.
//

#include "cutlass/gemm/volta884_shared_tile_contiguous.h"
#include "cutlass/gemm/volta884_shared_tile_crosswise.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Epilogue shared memory iterators
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Stores an accumulator fragment to shared memory
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_,
    /// Data type of mma.sync accumulator - this is used to infer layout.
    typename Accumulator_>
struct Volta884EpilogueSharedStoreIterator;

/// Loads an accumulator fragment from shared memory
template <
    /// Shape of warp-level GEMM
    typename WarpGemmTile_,
    /// Tiling of warp accumulator elements
    typename WarpDelta_,
    /// Data type of accumulator elements
    typename Scalar_,
    /// Number of scalar elements loaded
    int AccessSize_,
    /// Data type of mma.sync accumulator - this is used to infer layout.
    typename Accumulator_>
struct Volta884EpilogueSharedLoadIterator;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

//
// Partially-specialized implementations extracted in separate header.
//

#include "cutlass/gemm/volta884_shared_tile_epilogue.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
