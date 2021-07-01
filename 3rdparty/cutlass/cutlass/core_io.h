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
    \brief Helpers for printing cutlass/core objects
*/

#pragma once

#include <iosfwd>
#include <typeinfo>

#include "cutlass/coord.h"
#include "cutlass/vector.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Rank>
std::ostream& operator<<(std::ostream& out, Coord<Rank> const& coord) {
  for (int i = 0; i < Rank; ++i) {
    out << (i ? ", " : "") << coord.idx[i];
  }
  return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to enable formatted printing of CUTLASS scalar types to an ostream
template <typename T>
struct ScalarIO {

  /// Value to print
  T value;

  /// Default ctor
  ScalarIO() { }

  /// Constructs from a value
  ScalarIO(T value): value(value) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Default printing to ostream
template <typename T>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<T> const &scalar) {
  return out << scalar.value;
}

/// Printing to ostream of int8_t as integer rather than character
template <>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<int8_t> const &scalar) {
  return out << int(scalar.value);
}

/// Printing to ostream of uint8_t as integer rather than character
template <>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<uint8_t> const &scalar) {
  return out << unsigned(scalar.value);
}

/// Printing to ostream of vector of 1b elements
template <>
inline std::ostream &operator<<(
  std::ostream &out, 
  ScalarIO<cutlass::Vector<cutlass::bin1_t, 32> > const &scalar) {

  for (int i = 0; i < 32; i++) {
    out << int(scalar.value[i]);
    out << ((i != 31) ? ", " : "");
  }
  return out;
}

/// Printing to ostream of vector of 4b signed integer elements
template <>
inline std::ostream &operator<<(
  std::ostream &out, 
  ScalarIO<cutlass::Vector<cutlass::int4_t, 8> > const &scalar) {

  for (int i = 0; i < 8; i++) {
    out << int(scalar.value[i]);
    out << ((i != 7) ? ", " : "");
  }
  return out;
}

/// Printing to ostream of vector of 4b unsigned integer elements
template <>
inline std::ostream &operator<<(
  std::ostream &out, 
  ScalarIO<cutlass::Vector<cutlass::uint4_t, 8> > const &scalar) {

  for (int i = 0; i < 8; i++) {
    out << unsigned(scalar.value[i]);
    out << ((i != 7) ? ", " : "");
  }
  return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
