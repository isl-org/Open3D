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
    \brief Constructs an iterator that owns two tile iterator instances
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/zip_tensor_ref.h"
#include "cutlass/zip_fragment.h"
#include "cutlass/util/pair.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs an iterator from a pair of iterators
template <typename First_, typename Second_>
class ZipTileIterator {
 public:
  /// First iterator type
  typedef First_ First;

  /// Second iterator type
  typedef Second_ Second;
  
  ///
  typedef typename First::Scalar Scalar;

  /// Params object
  struct Params {
    /// Parameters of first iterator
    typename First::Params first;

    /// Parameters of second iterator
    typename Second::Params second;

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Constructs a parameters object
    CUTLASS_HOST_DEVICE
    Params(typename First::Params const &_first, typename Second::Params const &_second)
        : first(_first), second(_second) {}
  };

  /// Fragment type
  typedef ZipFragment<typename First::Fragment, typename Second::Fragment> Fragment;

  /// Predicate vector
  typedef typename First::PredicateVector PredicateVector;

  /// Index type
  typedef platform::Pair<typename First::Index, typename Second::Index> Index;

  /// Long index type
  typedef platform::Pair<typename First::LongIndex, typename Second::LongIndex> LongIndex;

  /// Tensor reference
  typedef ZipTensorRef<
    typename First::TensorRef,
    typename Second::TensorRef> TensorRef;

  //
  // Data members
  //

  /// First iterator
  First first;

  /// Second iterator
  Second second;

  //
  // Methods
  //

  /// Default constructor
  CUTLASS_DEVICE
  ZipTileIterator() {}

  /// Constructs a zip iterator from params
  CUTLASS_DEVICE
  ZipTileIterator(Params const &_params, Coord<3> const &threadblock_offset = make_Coord(0, 0, 0))
      : first(_params.first, threadblock_offset), second(_params.second, threadblock_offset) {}

  /// Constructs a zip iterator from iterator instances
  CUTLASS_DEVICE
  ZipTileIterator(First const &_first, Second const &_second) : first(_first), second(_second) {}

  /// Constructs a zip iterator from iterator instances
  CUTLASS_DEVICE
  ZipTileIterator(TensorRef const &ref) : first(ref.first), second(ref.second) {}

  /// Constructs a zip iterator from iterator instances
  CUTLASS_DEVICE
  ZipTileIterator(Params const &_params, TensorRef const &ref):
    first(_params.first, ref.first), second(_params.second, ref.second) {}

  //
  // Predicate initialization
  //

  /// Initializes a predicate vector using a RegularTilePredicateFunctor
  template <
      /// Predicate iterator
      typename PredicateIterator>
  CUTLASS_HOST_DEVICE void initialize_predicates(PredicateIterator predicate_it,
                                                 Coord<3> const &bounds,
                                                 Coord<3> const &block_offset = make_Coord(0,
                                                                                           0,
                                                                                           0)) {
    first.initialize_predicates(predicate_it, bounds, block_offset);
  }

  /// Initializes a predicate vector using an arbitrary predicate functor
  template <
      /// Predicate iterator
      typename PredicateIterator,
      /// Functor computing predicates
      typename PredicateFunctor>
  CUTLASS_HOST_DEVICE void initialize_predicates(PredicateIterator predicate_it,
                                                 PredicateFunctor const &functor,
                                                 Coord<3> const &block_offset) {
    first.initialize_predicates(predicate_it, functor, block_offset);
  }

  //
  // No predicates
  //

  /// Loads a fragment and increments without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void load_post_increment(Fragment &fragment) {
    first.load_post_increment(fragment.first);
    second.load_post_increment(fragment.second);
  }

  /// Loads a fragment and increments without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void load_post_increment(Fragment &fragment,
                            Coord<4> const &offset) {
    first.load_post_increment(fragment.first, offset);
    second.load_post_increment(fragment.second, offset);
  }

  /// Loads a fragment without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void load(Fragment &fragment) const {
    first.load(fragment.first);
    second.load(fragment.second);
  }

  /// Loads a fragment without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void load(Fragment &fragment,
                            Coord<4> const &offset) const {
    first.load(fragment.first, offset);
    second.load(fragment.second, offset);
  }

  /// Stores a fragment and increments without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void store_post_increment(Fragment const &fragment) {
    first.store_post_increment(fragment.first);
    second.store_post_increment(fragment.second);
  }

  /// Stores a fragment and increments without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void store_post_increment(Fragment const &fragment,
                            Coord<4> const &offset) {
    first.store_post_increment(fragment.first, offset);
    second.store_post_increment(fragment.second, offset);
  }

  /// Stores a fragment without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void store(Fragment const &fragment) const {
    first.store(fragment.first);
    second.store(fragment.second);
  }

  /// Stores a fragment without predicates
  template <typename Fragment>
  CUTLASS_DEVICE void store(Fragment const &fragment,
                            Coord<4> const &offset) const {
    first.store(fragment.first, offset);
    second.store(fragment.second, offset);
  }

  //
  // With predication
  //

  /// Loads a fragment and increments, using predicates
  template <typename Fragment, typename PredicateIterator>
  CUTLASS_DEVICE void load_post_increment(Fragment &fragment, PredicateIterator pred_it) {
    first.load_post_increment(fragment.first, pred_it);
    second.load_post_increment(fragment.second, pred_it);
  }

  /// Loads a fragment with predicates
  template <typename Fragment, typename PredicateIterator>
  CUTLASS_DEVICE void load(Fragment &fragment, PredicateIterator pred_it) const {
    first.load(fragment.first, pred_it);
    second.load(fragment.second, pred_it);
  }

  /// Loads a fragment and increments, using predicates
  template <typename Fragment, typename PredicateIterator>
  CUTLASS_DEVICE void store_post_increment(Fragment const &fragment, PredicateIterator pred_it) {
    first.store_post_increment(fragment.first, pred_it);
    second.store_post_increment(fragment.second, pred_it);
  }

  /// Loads a fragment with predicates
  template <typename Fragment, typename PredicateIterator>
  CUTLASS_DEVICE void store(Fragment const &fragment, PredicateIterator pred_it) const {
    first.store(fragment.first, pred_it);
    second.store(fragment.second, pred_it);
  }

  //
  // Advances the iterators
  //

  /// Increments store iterator to next tile
  CUTLASS_DEVICE ZipTileIterator &increment(int count = 1) {
    first.increment(count);
    second.increment(count);
    return *this;
  }

  /// Increments to next tile
  CUTLASS_DEVICE ZipTileIterator &operator++() { return increment(); }

  CUTLASS_DEVICE ZipTileIterator &operator+=(int count) { return increment(count); }

  /// Adds a vector offset to the underlying iterators
  CUTLASS_DEVICE ZipTileIterator &operator+=(Coord<3> const &offset) {
    first += offset;
    second += offset;
    return *this;
  }

  /// Increments store iterator to previous tile
  CUTLASS_DEVICE ZipTileIterator &decrement(int count = 1) {
    first.decrement(count);
    second.decrement(count);
    return *this;
  }

  /// Increments to subsequent tile
  CUTLASS_DEVICE ZipTileIterator &operator--() { return decrement(); }

  /// Decrements to previous tile
  CUTLASS_DEVICE ZipTileIterator &operator-=(int count) { return decrement(count); }

  /// Adds an offset to both iterators
  CUTLASS_DEVICE void add_pointer_offset(LongIndex offset) {
    first.add_pointer_offset(offset.first);
    second.add_pointer_offset(offset.second);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namspace cutlass
