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
    \brief Defines a type for restructuring a tile.
*/
#pragma once

#include "cutlass/shape.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The following functor reshapes a tile of data. The goal is to have at least kAccessSize in
// the inner-most dimension. If the user respects that constraint, there is nothing to be done. If
// that's not the case, this functor will correct that and "extract" the right number of elements
// from the next dimension.

template <typename Tile_, int kAccessSize_, bool = (Tile_::kC < kAccessSize_)>
struct ReshapeTile {
  typedef Tile_ Tile;
};

template <typename Tile_, int kAccessSize_>
struct ReshapeTile<Tile_, kAccessSize_, true> {
  // Make sure the W dimension of the tile is large enough.
  static_assert(Tile_::kW >= kAccessSize_, "The W dimension is too small");
  // Make sure the dimension can be divided by the number of scalars.
  static_assert(Tile_::kW % kAccessSize_ == 0, "Not supported");
  // Collapse the W dimension.
  typedef Shape<Tile_::kD, Tile_::kH, Tile_::kW / kAccessSize_, kAccessSize_> Tile;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Tile_, int kAccessSize_, int kLdsPerAccess_, bool = (Tile_::kC < (kAccessSize_ * kLdsPerAccess_))>
struct WmmaReshapeTile {
  typedef Tile_ Tile;
};

template <typename Tile_, int kAccessSize_, int kLdsPerAccess_>
struct WmmaReshapeTile<Tile_, kAccessSize_, kLdsPerAccess_, true> {
  // Make sure the W dimension of the tile is large enough.
  static_assert(Tile_::kW >= (kAccessSize_ * kLdsPerAccess_), "The W dimension is too small");
  // Make sure the dimension can be divided by the number of scalars.
  static_assert(Tile_::kW % (kAccessSize_ * kLdsPerAccess_) == 0, "Not supported");
  // Collapse the W dimension.
  typedef Shape<Tile_::kD, Tile_::kH, Tile_::kW / (kAccessSize_ * kLdsPerAccess_), (kAccessSize_ * kLdsPerAccess_)> Tile;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
