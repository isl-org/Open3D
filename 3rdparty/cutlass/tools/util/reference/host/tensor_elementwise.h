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
/* \file
  \brief Defines host-side elementwise operations on TensorView.
*/

#pragma once

// Standard Library includes
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <cstdlib>
#include <cmath>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "tools/util/distribution.h"
#include "tools/util/type_traits.h"
#include "tools/util/reference/host/tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random uniform distribution
template <typename View_>
struct RandomUniformFunc {

  /// View type
  typedef View_ View;

  /// Scalar type
  typedef typename View::Storage T;

  /// Coordinate in tensor's index space
  typedef typename View::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    /// View object
    View view;

    /// RNG seed
    unsigned seed;

    /// Distriubtion
    Distribution dist;

    /// Default ctor
    Params() { }

    /// Constructor
    Params(
      View const &view,
      unsigned seed,
      Distribution dist
    ): view(view), seed(seed), dist(dist) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  RandomUniformFunc(Params const &params): params(params) {
    std::srand(params.seed);
  }

  /// Compute random value and update RNG state
  void operator()(TensorCoord const &coord) {

    double range = params.dist.uniform.max - params.dist.uniform.min;

    double rnd = double(std::rand()) / double(RAND_MAX);

    rnd = params.dist.uniform.min + range * rnd;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    T result;
    if (params.dist.int_scale >= 0) {
      rnd = double(int(rnd * double(1 << params.dist.int_scale)));
      result = T(rnd / double(1 << params.dist.int_scale));
    }
    else {
      result = T(rnd);
    }

    params.view.at(coord) = result;
  }
};

/// Computes a random Gaussian distribution
template <typename View_>
struct RandomGaussianFunc {

  /// View type
  typedef View_ View;

  /// Scalar type
  typedef typename View::Storage T;

  /// Coordinate in tensor's index space
  typedef typename View::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    /// View object
    View view;

    /// RNG seed
    unsigned seed;

    /// RNG distribution
    Distribution dist;

    /// Default ctor
    Params() { }

    /// Constructor
    Params(
      View const &view,
      unsigned seed,
      Distribution dist
    ): view(view), seed(seed), dist(dist) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// Constant PI
  double pi;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  RandomGaussianFunc(Params const &params): params(params) {
    pi = std::acos(-1);
  }

  /// Compute random value and update RNG state
  void operator()(TensorCoord const &coord) {

    // Box-Muller transform to generate random numbers with Normal distribution
    double u1 = double(std::rand()) / double(RAND_MAX);
    double u2 = double(std::rand()) / double(RAND_MAX);

    double rnd = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);

    // Scale according to Gaussian distribution parameters
    rnd = params.dist.gaussian.mean + params.dist.gaussian.stddev * rnd;

    T result;
    if (params.dist.int_scale >= 0) {
      rnd = double(int(rnd * double(1 << params.dist.int_scale)));
      result = T(rnd / double(1 << params.dist.int_scale));
    }
    else {
      result = T(rnd);
    }

    params.view.at(coord) = result;
  }
};

/// Computes a linear combination of each element
template <typename View_>
struct LinearCombinationFunc {

  /// View type
  typedef View_ View;

  /// Scalar type
  typedef typename View::Storage T;

  /// Coordinate in tensor's index space
  typedef typename View::TensorCoord TensorCoord;

  //
  // Data members
  //

  /// TensorView object
  View view;

  /// Delta
  Coord<View::kRank, double> delta;

  /// Offset
  double offset;

  //
  // Methods
  //

  /// Constructor
  LinearCombinationFunc(
    View const &view,
    Distribution dist
  ): view(view) {

    offset = dist.linear.offset;
    if (View::kRank >= 1) {
      delta[View::kRank - 1] = dist.linear.delta_column;
    }
    if (View::kRank >= 2) {
      delta[View::kRank - 2] = dist.linear.delta_row;
    }
    // Additional ranks have delta of zero
    for (int i = View::kRank - 2; i > 0; --i) {
      delta[i - 1] = 0;
    }
  }

  /// Compute linear combination
  void operator()(TensorCoord const &coord) {
    double result = offset;

    for (int i = 0; i < View::kRank; ++i) {
      result += delta[i] * double(coord[i]);
    }
    view.at(coord) = T(result);
  }
};

/// Returns 1 or 0 if the coordinate is along the tensor's diagonal
template <typename View_>
struct IdentityFunc {

  /// TensorView
  typedef View_ View;

  /// Scalar type
  typedef typename View::Storage T;

  /// Coordinate in tensor's index space
  typedef typename View::TensorCoord TensorCoord;

  //
  // Data members
  //

  /// View object
  View view;

  /// Default ctor
  IdentityFunc(View const &view): view(view) { }

  /// Computes an identity
  void operator()(TensorCoord const &coord) {
    bool equal = true;
    for (int i = 0; i < View::kRank; ++i) {
      if (coord[i] != coord[0]) {
        equal = false;
      }
    }
    view.at(coord) = equal ? T(1) : T(0);
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Initializes a tensor randomly or procedurally.
template <typename View>
void TensorInitialize(View const &view,
                      unsigned seed,
                      Distribution const &dist) {

  typedef typename View::Storage Scalar;

  switch (dist.kind) {
    case Distribution::Uniform:
    {
      typedef detail::RandomUniformFunc<View> Func;
      typedef typename Func::Params Params;

      TensorForEach<Func, View::kRank, Params>(
        view.size(),
        Params(view, seed, dist)
      );
    }
      break;
    case Distribution::Gaussian:
    {
      typedef detail::RandomGaussianFunc<View> Func;
      typedef typename Func::Params Params;

      TensorForEach<Func, View::kRank, Params>(
        view.size(),
        Params(view, seed, dist)
      );
    }
      break;
    case Distribution::Linear:
    {
      typedef detail::LinearCombinationFunc<View> Func;
      TensorForEach<Func, View::kRank, Func>(
        view.size(),
        Func(view, dist));
    }
      break;
    case Distribution::Identity:
    {
      typedef detail::IdentityFunc<View> Func;

      Func func(view);

      TensorForEach<Func, View::kRank, Func>(view.size(), func);
    }
      break;
    default:
      break;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Compares two tensor views of equal rank and dimension.
template <typename ViewL, typename ViewR>
struct TensorEqualsFunc {

  /// Storage type
  typedef typename ViewL::Storage T;

  /// Unsigned integer type of same size as View type
  typedef typename cutlass::TypeTraits<T>::unsigned_type UnsignedType;

  /// Coordinate in tensor's index space
  typedef typename ViewL::TensorCoord TensorCoord;

  /// Assertions
  static_assert(ViewL::kRank == ViewR::kRank,
    "Cannot compare tensors of different rank");

  //
  // Data members
  //

  /// View of left-hand-side tensor
  ViewL lhs;

  /// View of right-hand-side tensor
  ViewR rhs;

  /// Pointer to result scalar - only written with 0 if values are incorrect
  int *result;

  //
  // Methods
  //

  /// Constructor
  TensorEqualsFunc(ViewL const &lhs, ViewR const &rhs, int *result): lhs(lhs), rhs(rhs), result(result) { }

  /// Equality check
  void operator()(TensorCoord const &coord) {
    UnsignedType _lhs = reinterpret_cast<UnsignedType const &>(lhs.at(coord));
    UnsignedType _rhs = reinterpret_cast<UnsignedType const &>(rhs.at(coord));
    if (_lhs != _rhs) {
      *result = 0;
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two tensor views are equal.
template <typename ViewL, typename ViewR>
bool TensorEquals(ViewL const &lhs, ViewR const &rhs) {

  // Sizes must be identical
  if (lhs.size() != rhs.size()) {
    return false;
  }

  int result = 1;

  typedef detail::TensorEqualsFunc<ViewL, ViewR> Func;
  Func func(lhs, rhs, &result);

  TensorForEach<Func, ViewL::kRank, Func>(lhs.size(), func);

  return result != 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to apply a binary operator in place
template <typename ViewL, typename ViewR, typename BinaryFunc>
struct TensorFuncBinaryOp {

  /// Coordinate in tensor's index space
  typedef typename ViewL::TensorCoord TensorCoord;

  //
  // Data members
  //

  /// View of left-hand-side tensor
  ViewL lhs;

  /// View of right-hand-side tensor
  ViewR rhs;

  /// Binary function applied to each element
  BinaryFunc func;

  //
  // Methods
  //

  /// Constructor
  TensorFuncBinaryOp(
    ViewL const &lhs,
    ViewR const &rhs,
    BinaryFunc func = BinaryFunc()): lhs(lhs), rhs(rhs), func(func) { }

  /// Equality check
  void operator()(TensorCoord const &coord) {
    lhs.at(coord) = func(lhs.at(coord), rhs.at(coord));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
