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
    \brief Implements the tile stream concept, composing an iterator with a transformation. Offers
      split-phase semantics, separating the initiation of an asynchronous memory operation with a
      fence forcing it to complete.
*/
#pragma once

// clang-format off

#include "cutlass/convert.h"
#include "cutlass/tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic stream for loading and transforming fragments
template <typename Iterator_, typename Transformer_ = Copy<typename Iterator_::Fragment> >
struct TileLoadStream {
  //
  // Type definitions
  //

  /// TileLoadIterator
  typedef Iterator_ Iterator;

  /// Transformer
  typedef Transformer_ Transformer;

  /// Fragment fetched from source memory
  typedef typename Iterator::Fragment Fragment;

  /// Output fragment from transformer
  typedef typename Transformer::OutputFragment TransformedFragment;

  /// Tensor reference expected by the stream
  typedef typename Iterator::TensorRef TensorRef;

  /// Empty predicate vector struct
  struct PredicateVector {};

  /// Index type
  typedef typename Iterator::Index Index;

  /// Parameters object used to construct generic load stream
  struct Params {
    /// Parameters to the iterator
    typename Iterator::Params iterator;

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Constructor with iterator params
    CUTLASS_HOST_DEVICE
    Params(typename Iterator::Params const &_iterator) : iterator(_iterator) {}
  };

  //
  // Data members
  //

  /// Iterator to load tiles
  Iterator iterator;

  /// Fragment loaded via iterator
  Fragment fetched_fragment;

  /// Transformation applied to fragments
  Transformer transformer;

  /// Transformed fragment from transformer
  TransformedFragment transformed_fragment;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  TileLoadStream(Params const &_params, TensorRef const &_ref)
      : iterator(_params.iterator, _ref) {}

  /// Ctor
  CUTLASS_DEVICE
  TileLoadStream(Params const &_params,
    Coord<3> const &threadblock_offset = make_Coord(0, 0, 0)
  ): iterator(_params.iterator, threadblock_offset) { }

  /// Loads a tile and increments the iterator
  CUTLASS_DEVICE
  void copy() { iterator.load_post_increment(fetched_fragment); }

  /// Commits the fetched fragment and applies a transformation
  CUTLASS_DEVICE
  void commit() { transformer.transform(fetched_fragment, transformed_fragment); }

  /// Accesses the loaded, transformed fragment
  CUTLASS_DEVICE
  Fragment &intermediate_fragment() { return fetched_fragment; }

  /// Accesses the loaded, transformed fragment
  CUTLASS_DEVICE
  TransformedFragment &fragment() { return transformed_fragment; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic stream for transforming and storing fragments
template <typename Iterator_, typename Transformer_ = Copy<typename Iterator_::Fragment> >
struct TileStoreStream {
  //
  // Type definitions
  //

  /// TileLoadIterator
  typedef Iterator_ Iterator;

  /// Transformer
  typedef Transformer_ Transformer;

  /// Source fragment
  typedef typename Transformer::InputFragment Fragment;

  /// Transformed fragment, compatible with Iterator::Fragment
  typedef typename Transformer::OutputFragment TransformedFragment;

  /// Tensor reference expected by the underlying iterator
  typedef typename Iterator::TensorRef TensorRef;

  /// Empty predicate vector struct
  struct PredicateVector {};

  /// Index type
  typedef typename Iterator::Index Index;

  /// Parameters used to construct the stream
  struct Params {
    /// Parameters to the iterator
    typename Iterator::Params iterator;

    //
    // Methods
    //

    /// Default constructor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Constructor with iterator params
    CUTLASS_HOST_DEVICE
    Params(typename Iterator::Params const &_iterator) : iterator(_iterator) {}
  };

  //
  // Data members
  //

  /// Iterator to store tiles
  Iterator iterator;

  /// Transformation applied to inputs
  Transformer transformer;

  /// Source fragment
  Fragment source_fragment;

  /// Transformed fragment from transformer
  TransformedFragment transformed_fragment;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  TileStoreStream(Params const &_params, TensorRef const &_ref)
      : iterator(_params.iterator, _ref) {}

  /// Ctor
  CUTLASS_DEVICE
  TileStoreStream(Params const &_params,
                  Coord<3> const &threadblock_offset = make_Coord(0, 0, 0)
  ): iterator(_params.iterator, threadblock_offset) { }

  /// Stores a fragment and increments the iterator
  CUTLASS_DEVICE
  void copy() {

    transformer.transform(source_fragment, transformed_fragment);
    iterator.store_post_increment(transformed_fragment);
  }

  /// Stores a fragment and increments the iterator
  CUTLASS_DEVICE
  void copy(Fragment const &frag) {
    source_fragment = frag;
    copy();
  }

  /// Commits the store operation
  CUTLASS_DEVICE
  void commit() {}

  /// Accesses the transformed fragment
  CUTLASS_DEVICE
  Fragment &fragment() { return source_fragment; }

  /// Accesses the fragment after trasnforming
  CUTLASS_DEVICE
  TransformedFragment &intermediate_fragment() { return transformed_fragment; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic stream for loading and transforming fragments
template <typename Iterator_,
          typename PredicateFunctor_ =
              RegularTilePredicateFunctor<typename Iterator_::Traits::Delta>,
          typename Transformer_ = Copy<typename Iterator_::Fragment> >
struct PredicatedTileLoadStream : public TileLoadStream<Iterator_, Transformer_> {
  //
  // Type definitions
  //

  typedef TileLoadStream<Iterator_, Transformer_> Base;

  /// TileLoadIterator
  typedef Iterator_ Iterator;

  /// Predicate functor
  typedef PredicateFunctor_ PredicateFunctor;

  /// Transformer
  typedef Transformer_ Transformer;

  /// Fragment fetched from source memory
  typedef typename Base::Fragment Fragment;

  /// Output fragment from transformer
  typedef typename Base::TransformedFragment TransformedFragment;

  /// Parameters object used to construct generic load stream
  typedef typename Base::Params Params;
  
  ///
  typedef typename Iterator::Scalar Scalar;

  //
  // Data members
  //

  /// Predicates
  typename Iterator::PredicateVector predicates;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  PredicatedTileLoadStream(Params const &_params,
                          Coord<3> const &bounds,
                          Coord<3> const &threadblock_offset = make_Coord(0, 0, 0))
      : Base(_params, threadblock_offset) {
    this->iterator.initialize_predicates(
        predicates.begin(), PredicateFunctor(bounds), threadblock_offset);
  }

  /// Loads a tile and increments the iterator
  CUTLASS_DEVICE
  void copy() { this->iterator.load_post_increment(this->fetched_fragment, predicates.begin()); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic stream for transforming and storing fragments
template <typename Iterator_,
          typename PredicateFunctor_ =
              RegularTilePredicateFunctor<typename Iterator_::Traits::Delta>,
          typename Transformer_ = Copy<typename Iterator_::Fragment> >
struct PredicatedTileStoreStream : public TileStoreStream<Iterator_, Transformer_> {
  //
  // Type definitions
  //

  typedef TileStoreStream<Iterator_, Transformer_> Base;

  /// TileLoadIterator
  typedef Iterator_ Iterator;

  /// Predicate functor
  typedef PredicateFunctor_ PredicateFunctor;

  /// Transformer
  typedef Transformer_ Transformer;

  /// Fragment fetched from source memory
  typedef typename Base::Fragment Fragment;

  /// Output fragment from transformer
  typedef typename Base::TransformedFragment TransformedFragment;

  /// Parameters object used to construct generic load stream
  typedef typename Base::Params Params;

  ///
  typedef typename Iterator::Scalar Scalar;

  //
  // Data members
  //

  /// Predicates
  typename Iterator::PredicateVector predicates;

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  PredicatedTileStoreStream(Params const &_params,
                           Coord<3> const &bounds,
                           Coord<3> const &threadblock_offset = make_Coord(0, 0, 0))
      : Base(_params, threadblock_offset) {
    this->iterator.initialize_predicates(
        predicates.begin(), PredicateFunctor(bounds), threadblock_offset);
  }

  /// Stores the fragment and increments the iterator
  CUTLASS_DEVICE
  void copy() {
    this->transformer.transform(this->source_fragment, this->transformed_fragment);
    this->iterator.store_post_increment(this->transformed_fragment, predicates.begin());
  }

  /// Stores the fragment and increments the iterator
  CUTLASS_DEVICE
  void copy(Fragment const &frag) {
    this->source_fragment = frag;
    copy();
  }

  /// Commits the store operation
  CUTLASS_DEVICE
  void commit() {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

// clang-format on
