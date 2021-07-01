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
  \brief Defines device-side elementwise operations on TensorView. Note, the operations defined
    in this header are not specialized for any particular data layout and are therefore not
    intended to offer the best possible performance. Rather, they are intended to be generic
    reference implementations to support the CUTLASS unit tests.
*/

#pragma once

// Standard Library includes
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

// CUDA includes
#include <cublas_v2.h>
#include <curand_kernel.h>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "tools/util/device_memory.h"
#include "tools/util/distribution.h"
#include "tools/util/type_traits.h"
#include "tools/util/host_tensor.h"
#include "tools/util/reference/device/tensor_foreach.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace reference {
namespace device {

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
    int64_t seed;

    /// Distriubtion
    Distribution dist;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Constructor
    CUTLASS_HOST_DEVICE
    Params(
      View const &view,
      int64_t seed,
      Distribution dist
    ): view(view), seed(seed), dist(dist) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomUniformFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    double range = params.dist.uniform.max - params.dist.uniform.min;
    double rnd = curand_uniform(&rng_state);
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
    int64_t seed;

    /// RNG distribution
    Distribution dist;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Constructor
    CUTLASS_HOST_DEVICE
    Params(
      View const &view,
      int64_t seed,
      Distribution dist
    ): view(view), seed(seed), dist(dist) { }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomGaussianFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    double rnd = curand_normal(&rng_state);
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
  CUTLASS_HOST_DEVICE
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
  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {
    double result = offset;
    CUTLASS_PRAGMA_UNROLL
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
  CUTLASS_HOST_DEVICE
  IdentityFunc(View const &view): view(view) { }

  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {
    bool equal = true;
    CUTLASS_PRAGMA_UNROLL
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
                      int64_t seed,
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

} // namespace device
} // namespace reference

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to appropriate initialization kernel - preserved for backwards compatibility
template <typename T>
inline void tensor_initialize(Distribution const &dist,
                              int64_t seed,
                              int dim_contiguous,
                              int dim_strided,
                              T *tensor,
                              int ldm) {

  TensorView<T, 2> view(tensor, make_Coord(ldm, 1), make_Coord(dim_strided, dim_contiguous));
  reference::device::TensorInitialize(view, seed, dist);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace reference {
namespace device {
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
  CUTLASS_HOST_DEVICE
  TensorEqualsFunc(ViewL const &lhs, ViewR const &rhs, int *result): lhs(lhs), rhs(rhs), result(result) { }

  /// Equality check
  CUTLASS_HOST_DEVICE
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

  // Allocate device memory to contain result of kernel reduction
  HostTensor<int, 1> result(1);
  result.fill(1);
  result.sync_device();

  typedef detail::TensorEqualsFunc<ViewL, ViewR> Func;
  Func func(lhs, rhs, result.device_data());

  TensorForEach<Func, ViewL::kRank, Func>(lhs.size(), func);
  result.sync_host();

  return result.at(0) != 0;
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
  CUTLASS_HOST_DEVICE
  TensorFuncBinaryOp(
    ViewL const &lhs,
    ViewR const &rhs,
    BinaryFunc func = BinaryFunc()): lhs(lhs), rhs(rhs), func(func) { }

  /// Equality check
  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {
    lhs.at(coord) = func(lhs.at(coord), rhs.at(coord));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to apply a binary operator in place
template <typename ViewL, typename ViewR>
struct TensorFillFunc {

  /// Coordinate in tensor's index space
  typedef typename ViewL::TensorCoord TensorCoord;

  /// Destination element type
  typedef typename ViewL::Storage DestType;

  /// Source element type
  typedef typename ViewR::Storage SrcType;

  /// Parameters object
  struct Params {

    //
    // Data members
    //

    /// View of left-hand-side tensor
    ViewL lhs;

    /// View of right-hand-side tensor
    ViewR rhs;

    /// Source offset coordinate
    TensorCoord source_offset;

    /// Size of the subtensor copied from the source
    TensorCoord source_size;

    /// Offset in destination
    TensorCoord dest_offset;

    //
    // Methods
    //

    /// Constructs a parameters object for filling a tensor
    Params(
      ViewL const &lhs,
      ViewR const &rhs,
      TensorCoord const &source_offset = TensorCoord()
    ):
      lhs(lhs), rhs(rhs), source_offset(source_offset), source_size(rhs.size() - source_offset) { }

    /// Constructs a parameters object for filling a tensor
    Params(
      ViewL const &lhs,
      ViewR const &rhs,
      TensorCoord const &source_offset,
      TensorCoord const &source_size,
      TensorCoord const &dest_offset = TensorCoord()
    ):
      lhs(lhs), rhs(rhs), source_offset(source_offset), source_size(source_size), dest_offset(dest_offset) { }
  };

  //
  // Data members
  //

  Params params;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorFillFunc(
    Params const &params): params(params) { }

  /// Equality check
  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {

    TensorCoord dst_coord = params.dest_offset + coord;
    TensorCoord src_coord = params.source_offset + coord;

    if (dst_coord < params.lhs.size() && src_coord < params.rhs.size()) {
      params.lhs.at(dst_coord) = DestType(params.rhs.at(src_coord));
    }
  }
};

} // namespace detail

/// Fills a TensorView with the elements from another TensorView
template <typename ViewL, typename ViewR>
void TensorFill(
  ViewL lhs,
  ViewR rhs,
  typename ViewL::TensorCoord const &source_offset,
  typename ViewL::TensorCoord const &source_size,
  typename ViewL::TensorCoord const &dest_offset) {

  typedef typename ViewL::TensorCoord TensorCoord;

  TensorCoord dst_size = lhs.size() - dest_offset;
  TensorCoord src_size = rhs.size() - source_offset;

  TensorCoord fill_size = dst_size.clamp(src_size);

  // Fill function
  typedef detail::TensorFillFunc<ViewL, ViewR> Func;
  typedef typename Func::Params Params;

  Params params(lhs, rhs, source_offset, source_size, dest_offset);

  TensorForEach<Func, ViewL::kRank, Params>(fill_size, params);
}

/// Fills a TensorView with the elements from another TensorView
template <typename ViewL, typename ViewR>
void TensorFill(
  ViewL lhs,
  ViewR rhs,
  typename ViewL::TensorCoord const &source_offset = typename ViewL::TensorCoord()) {

  typedef typename ViewL::TensorCoord TensorCoord;

  TensorFill(lhs, rhs, source_offset, rhs.size(), TensorCoord());
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to apply a binary operator in place
template <typename ViewL>
struct TensorFillElementFunc {

  /// Coordinate in tensor's index space
  typedef typename ViewL::TensorCoord TensorCoord;

  /// Destination element type
  typedef typename ViewL::Storage DestType;

  /// Parameters object
  struct Params {

    //
    // Data members
    //

    /// View of left-hand-side tensor
    ViewL lhs;

    /// Source offset coordinate
    TensorCoord offset;

    /// Element to overwrite with
    DestType value;

    //
    // Methods
    //

    /// Constructs a parameters object for filling a tensor
    CUTLASS_HOST_DEVICE
    Params(
      ViewL const &lhs,
      DestType const &value,
      TensorCoord const &offset = TensorCoord()
    ):
      lhs(lhs), value(value), offset(offset) { }
  };

  //
  // Data members
  //

  Params params;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  TensorFillElementFunc(
    Params const &params): params(params) { }

  /// Equality check
  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {

    TensorCoord dst_coord = params.offset + coord;

    if (dst_coord < params.size) {
      params.lhs.at(dst_coord) = params.value;
    }
  }
};

} // namespace detail

/// Method to perform the actual fill
template <typename ViewL>
void TensorFillElement(
  ViewL const &lhs,
  typename ViewL::Storage const &value,
  typename ViewL::TensorCoord const &offset,
  typename ViewL::TensorCoord const &size) {

  // Fill function
  typedef detail::TensorFillElementFunc<ViewL> Func;
  typedef typename Func::Params Params;

  Params params(lhs, value, offset);

  TensorForEach<Func, ViewL::kRank, Params>(size, params);
}

/// Fills a tensor
template <typename ViewL>
void TensorFillElement(
  ViewL lhs,
  typename ViewL::Storage value,
  typename ViewL::TensorCoord const &offset =typename ViewL::Storage()) {

  TensorFillElement(lhs, value, offset, lhs.size() - offset);
}

/// Constructs a parameters object for filling a tensor
template <typename ViewL>
void TensorFillElement(
  ViewL lhs,
  typename ViewL::Storage value,
  typename ViewL::Storage const &offset,
  typename ViewL::Storage const &size) {

  TensorFillElement(lhs, value, offset, size);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass

