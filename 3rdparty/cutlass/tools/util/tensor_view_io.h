/***************************************************************************************************
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/core_io.h"
#include "cutlass/tensor_view.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to write the least significant rank of a TensorView
template <
  typename Storage_,
  int Rank_,
  typename MapFunc_,
  int StorageRank_,
  typename Index_,
  typename LongIndex_
>
inline std::ostream & TensorView_WriteLeastSignificantRank(
  std::ostream& out, 
  cutlass::TensorView<
    Storage_, 
    Rank_, 
    MapFunc_, 
    StorageRank_, 
    Index_, 
    LongIndex_> const& tensor,
  cutlass::Coord<Rank_> const &start_coord,
  int rank,
  std::streamsize width) {

  for (int idx = 0; idx < tensor.size(rank); ++idx) {

    Coord<Rank_> coord(start_coord);
    coord[rank] = idx;

    if (idx) {
      out.width(0);
      out << ", ";
    }
    if (idx || coord) {
      out.width(width);
    }
    out << ScalarIO<Storage_>(tensor.at(coord));
  }

  return out;
}

/// Helper to write a rank of a TensorView
template <
  typename Storage_,
  int Rank_,
  typename MapFunc_,
  int StorageRank_,
  typename Index_,
  typename LongIndex_
>
inline std::ostream & TensorView_WriteRank(
  std::ostream& out, 
  cutlass::TensorView<
    Storage_, 
    Rank_, 
    MapFunc_, 
    StorageRank_, 
    Index_, 
    LongIndex_> const& tensor,
  cutlass::Coord<Rank_> const &start_coord,
  int rank,
  std::streamsize width) {

  // If called on the least significant rank, write the result as a row
  if (rank + 1 == Rank_) {
    return TensorView_WriteLeastSignificantRank(out, tensor, start_coord, rank, width);
  }

  // Otherwise, write a sequence of rows and newlines
  for (int idx = 0; idx < tensor.size(rank); ++idx) {

    Coord<Rank_> coord(start_coord);
    coord[rank] = idx;

    if (rank + 2 == Rank_) {
      // Write least significant ranks asa matrix with rows delimited by ";\n"
      out << (idx ? ";\n" : "");
      TensorView_WriteLeastSignificantRank(out, tensor, coord, rank + 1, width);
    }
    else {
      // Higher ranks are separated by newlines
      out << (idx ? "\n" : "");
      TensorView_WriteRank(out, tensor, coord, rank + 1, width);
    }
  }

  return out;
}

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints human-readable representation of a TensorView to an ostream
template <
  typename Storage_,
  int Rank_,
  typename MapFunc_,
  int StorageRank_,
  typename Index_,
  typename LongIndex_
>
inline std::ostream& operator<<(
  std::ostream& out, 
  TensorView<
    Storage_, 
    Rank_, 
    MapFunc_, 
    StorageRank_, 
    Index_, 
    LongIndex_> const& tensor) {

  // Prints a TensorView according to the following conventions:
  //   - least significant rank is printed as rows separated by ";\n"
  //   - all greater ranks are delimited with newlines
  //
  // The result is effectively a whitespace-delimited series of 2D matrices.

  return detail::TensorView_WriteRank(out, tensor, Coord<Rank_>(), 0, out.width());
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
