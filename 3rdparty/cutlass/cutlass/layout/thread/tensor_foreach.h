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
#pragma once

//#include <stdexcept>
#include "cutlass/cutlass.h"
//#include "tools/util/reference/device/kernel/tensor_foreach.h"

namespace cutlass {
namespace layout {
namespace thread {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines several helpers
namespace detail {

/// Helper to perform for-each operation
template <typename Func, int Rank, int RankRemaining>
struct TensorForEachHelper {
  /// Index of the active rank
  static int const kActiveRank = Rank - RankRemaining - 1;

  /// Constructor for general rank
  CUTLASS_DEVICE TensorForEachHelper(Func &func, Coord<Rank> const &size, Coord<Rank> &coord) {
    for (int i = 0; i < size.at(kActiveRank); ++i) {
      coord[kActiveRank] = i;
      TensorForEachHelper<Func, Rank, RankRemaining - 1>(func, size, coord);
    }
  }
};

/// Helper to perform for-each operation
template <typename Func, int Rank>
struct TensorForEachHelper<Func, Rank, 0> {
  /// Index of the active rank
  static int const kActiveRank = Rank - 1;

  /// Constructor for fastest chaning rank
  CUTLASS_DEVICE TensorForEachHelper(Func &func, Coord<Rank> const &size, Coord<Rank> &coord) {
    for (int i = 0; i < size.at(kActiveRank); ++i) {
      coord[kActiveRank] = i;
      func(coord);
    }
  }
};

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterates over the index space of a tensor
template <typename Func, int Rank, typename Params>
struct TensorForEach {
  /// Constructor performs the operation.
  CUTLASS_DEVICE TensorForEach(Coord<Rank> size, Params params = Params()) {
    Func func(params);
    Coord<Rank> coord;

    detail::TensorForEachHelper<Func, Rank, Rank - 1>(func, size, coord);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace layout
}  // namespace cutlass
