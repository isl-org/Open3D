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
    \brief Defines abstractions for managing loading and storing fragments to shared memory in the
      efficient GEMM pipeline.
*/
#pragma once

#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/gemm_shared_tile.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The load iterator.
    typename Iterator_,
    /// The transformer to be applied after the data has been copied from shared memory.
    typename Transformer_ = Copy<typename Iterator_::Fragment> >

struct SharedLoadStream {
  /// The load iterator.
  typedef Iterator_ Iterator;
  /// The transformer.
  typedef Transformer_ Transformer;

  /// The fragment that is copied from shared memory.
  typedef typename Iterator::Fragment FetchedFragment;
  /// The fragment that is obtained after the transformation by the transformer.
  typedef typename Transformer::OutputFragment TransformedFragment;
  /// Make sure the fragments match.
  static_assert((platform::is_same<FetchedFragment, typename Transformer::InputFragment>::value),
                "");
  /// The output fragment.
  typedef TransformedFragment Fragment;
  /// Scalar data type
  typedef typename Iterator::Scalar Scalar;

  /// Reference type to a tensor
  typedef TensorRef<Scalar, 4> TensorRef;

  /// The params.
  struct Params {
    /// The iterator params.
    typename Iterator::Params iterator;

    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize() { return iterator.initialize(); }
  };

  /// The storage in shared memory needed by that stream.
  typedef typename Iterator::Storage SharedStorage;

  /// Ctor.
  CUTLASS_DEVICE SharedLoadStream() {}

  /// Ctor.
  CUTLASS_DEVICE SharedLoadStream(Params const &params, TensorRef const &ref) {
    this->initialize(params, ref);
  }

  /// Initialize the stream.
  CUTLASS_DEVICE void initialize(Params const &params, TensorRef const &ref) {
    // The iterator.
    iterator = Iterator(params.iterator, ref.data());
    // The transformer.
    transformer = Transformer();
  }

  /// Clears the fragment
  CUTLASS_DEVICE void clear() {
    fetched[0].clear();
    fetched[1].clear();
    transformed[0].clear();
    transformed[1].clear();
  }

  /// Load the data from shared memory to the fetch fragment.
  CUTLASS_DEVICE void copy() {
    iterator.load_post_increment(fetched[0]);
  }

  /// Load the data from shared memory to the fetch fragment.
  CUTLASS_DEVICE void copy(int step) { iterator.load(fetched[step % 2], step); }

  /// Commit the data.
  CUTLASS_DEVICE void commit() { transformer.transform(fetched[0], transformed[0]); }

  /// Commit the data.
  CUTLASS_DEVICE void commit(int step) {
    transformer.transform(fetched[step % 2], transformed[step % 2]);
  }

  /// Returns the fragment for the given step
  CUTLASS_DEVICE TransformedFragment &fragment(int step = 0) { return transformed[step % 2]; }

  /// Returns the fragment for the given step
  CUTLASS_DEVICE TransformedFragment const &fragment(int step = 0) const {
    return transformed[step % 2];
  }

  /// Increment the stage.
  CUTLASS_DEVICE void inc_stage() { iterator.inc_stage(); }

  /// The iterator.
  Iterator iterator;
  /// Fetched fragment
  FetchedFragment fetched[2];
  /// The transformer.
  Transformer transformer;
  /// Transformed fragment
  TransformedFragment transformed[2];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
