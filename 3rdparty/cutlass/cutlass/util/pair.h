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
    \brief Defines a pair<>
*/

#pragma once

namespace cutlass {
namespace platform {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs an iterator from a pair of iterators
template <typename T1, typename T2>
struct Pair {

  typedef T1 first_type;
  typedef T2 second_type;

  //
  // Data members
  //

  T1 first;
  T1 second;

  //
  // Methods
  //

  /// Default constructor
  CUTLASS_HOST_DEVICE
  Pair() { }

  /// Constructs a pair
  CUTLASS_HOST_DEVICE
  Pair(T1 const &first_, T2 const &second_): first(first_), second(second_) { }
};

/// Constructs a pair and deduces types
template <typename T1, typename T2>
Pair<T1, T2> make_Pair(T1 const &first, T2 const &second) {
  return Pair<T1, T2>(first, second);
}

/// Equality
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator==(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  return (lhs.first == rhs.first) && (lhs.second == rhs.second);
}

/// Inequality
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator!=(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  return !(lhs == rhs);
}

/// Lexical comparison
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator<(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  if (lhs.first < rhs.first) {
    return true;
  }
  else if (rhs.first < lhs.first) {
    return false;
  }
  else if (rhs.second < rhs.second) {
    return false;
  }
  return false;
}

/// Lexical comparison
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator<=(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  return !(rhs < lhs);
}

/// Lexical comparison
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator>(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  return (rhs < lhs);
}

/// Lexical comparison
template <typename T1, typename T2>
CUTLASS_HOST_DEVICE
bool operator>=(Pair<T1,T2> const &lhs, Pair<T1,T2> const &rhs) {
  return !(lhs < rhs);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace platform
} // namespace cutlass
