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
    \brief Tile traits used to construct global tile iterator for HGEMM. This is intended to
      partition the thread block-level tile into 2D subtiles loaded by the threads and facilitate
      memory accesses larger than 16 bits.
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/gemm/gemm_global_tile.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/reshape_tile.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GemmOperand::Kind kOperand_,
          MatrixLayout::Kind kLayout_,
          typename Scalar_,
          typename Tile_,
          typename Threads_,
          int kAccessSize_>
struct HgemmCrosswiseGlobalTileTraits : public GemmGlobalTileTraits<
                                            // Which GEMM operand?
                                            kOperand_,
                                            // The layout.
                                            kLayout_,
                                            // The scalar.
                                            Scalar_,
                                            // The tile.
                                            Tile_,
                                            // The threads.
                                            Threads_,
                                            // The number of scalars per LDG/STG.
                                            kAccessSize_> {
  /// The base class.
  typedef GemmGlobalTileTraits<kOperand_, kLayout_, Scalar_, Tile_, Threads_, kAccessSize_> Base;
  /// The threads.
  typedef typename Base::Threads Threads;
  /// The threads strides.
  typedef Shape<1, 2, Base::VectorizedTile::kC> ThreadsDelta;
  /// The strides in each dimension between different loads/stores.
  typedef Shape<Base::Threads::kH * 2, 1, Base::Threads::kW, Base::kAccessSize> Delta;
  /// The number of iterations needed to load/store the tile.
  typedef Shape<Base::VectorizedTile::kH / Base::Threads::kH / 2,
                2,
                Base::VectorizedTile::kW / Base::Threads::kW,
                Base::VectorizedTile::kC / Base::kAccessSize>
      Iterations;
  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    Coord<4> operator()() const {
      int thread_offset_h = threadIdx.x / Threads::kW * ThreadsDelta::kH;
      int thread_offset_w = threadIdx.x % Threads::kW * ThreadsDelta::kW;

      return make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
