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
    \brief Defines a coordinate used for the CUTLASS 4-D tile structure. 
*/

#pragma once

#include "cutlass/coord.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// TileCoord wraps Coord<4, int> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=4 index space of a CUTLASS tile structure should use TileCoord. 
template <typename Index_ = int>
struct TileCoord : public Coord<4, Index_> {
  
  /// Index type
  typedef Index_ Index;

  /// Underlying Coord<4>
  typedef Coord<4, Index> Base;

  /// D dimension
  static int kD = 0;

  /// H dimension
  static int kH = 1;

  /// W dimension
  static int kW = 2;

  /// C dimension
  static int kC = 3;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  TileCoord() { }

  /// Constructs from Coord<3> and infers coord[kC] = 0
  CUTLASS_HOST_DEVICE
  TileCoord(Coord<3, Index> const &coord): 
    Base(make_Coord(coord[0], coord[1], coord[2], 0)) { }

  /// Constructs from Coord<4>
  CUTLASS_HOST_DEVICE
  TileCoord(Coord<4, Index> const &coord): Base(coord) { }

  /// Constructs from an array of coordinate elements
  CUTLASS_HOST_DEVICE
  TileCoord(Index coord[4]): Base(coord) { }
  
  /// Helper to construct from a row and column
  CUTLASS_HOST_DEVICE
  TileCoord(Index d, Index h, Index w, Index c): Base(make_Coord(d, h, w, c)) { }

  /// Returns the D element of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & d() const { return this->at(kD); }

  /// Returns the D element of the coordinate
  CUTLASS_HOST_DEVICE
  Index & d() { return this->at(kD); }

  /// Returns the H element of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  /// Returns the H element of the coordinate
  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  /// Returns the W element of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  /// Returns the W element of the coordinate
  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  /// Returns the Celement of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & c() const { return this->at(kC); }

  /// Returns the C element of the coordinate
  CUTLASS_HOST_DEVICE
  Index & c() { return this->at(kC); }

  /// Gets H and W dimensions as a Coord<2>
  CUTLASS_HOST_DEVICE
  Coord<2> hw() const {
    return make_Coord(h(), w());
  }

  /// Gets H, W, and C dimensions as a Coord<3>
  CUTLASS_HOST_DEVICE
  Coord<3> hwc() const {
    return make_Coord(h(), w(), c());
  }

  /// Gets D, H, and W dimensions as a Coord<3>
  CUTLASS_HOST_DEVICE
  Coord<3> dhw() const {
    return make_Coord(d(), h(), w());
  }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  TileCoord operator+(Base const& b) const {
    return TileCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  TileCoord operator-(Base const& b) const {
    return TileCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  TileCoord operator*(Base const& b) const {
    return TileCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  TileCoord operator/(Base const& b) const {
    return TileCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  TileCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  TileCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  TileCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  TileCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
