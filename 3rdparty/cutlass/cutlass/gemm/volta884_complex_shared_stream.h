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
    \brief Implements efficient loading of the thread block-level tile from global memory and
   storing to shared memory.
*/

#pragma once

#include "cutlass/convert.h"
#include "cutlass/zip_fragment.h"
#include "cutlass/zip_tensor_ref.h"
#include "cutlass/zip_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Stream from shared memory to fragments for warp-level matrix multiply-accumulate
template <
    /// The load iterator.
    typename Iterator_,
    /// The transformer to be applied after the data has been copied from shared memory.
    typename Transformer_ = Copy<typename Iterator_::Fragment>,
    /// Number of increments before iterator wraps - zero indicates no wrapping
    int StageCount = 1>
struct Volta884ComplexSharedLoadStream {
  /// The load iterator.
  typedef Iterator_ RealIterator;

  /// Zips two real-valued iterators together
  typedef ZipTileIterator<RealIterator, RealIterator> Iterator;

  /// The transformer.
  typedef Transformer_ RealTransformer;

  /// Zips two transfoerms
  typedef ZipConvert<RealTransformer, RealTransformer> Transformer;

  /// Number of increments before iterator wraps - zero indicates no wrapping
  static int const kStageCount = StageCount;

  /// The fragment that is copied from shared memory.
  typedef typename Iterator::Fragment FetchedFragment;

  /// The fragment that is obtained after the transformation by the transformer.
  typedef typename Transformer::OutputFragment TransformedFragment;

  /// Make sure the fragments match.
  static_assert((platform::is_same<FetchedFragment, typename Transformer::InputFragment>::value),
                "");

  /// The output fragment.
  typedef TransformedFragment Fragment;

  /// Reference type
  typedef ZipTensorRef<
    TensorRef<half, 4>,
    TensorRef<half, 4>
  > TensorRef;

  /// Parameters passed from host
  struct Params { };

  //
  // Data members
  //

  /// Iterator for loading fragments for warp-level matrix multiply-accumulate
  Iterator iterator;

  /// Fetched fragment
  FetchedFragment fetched[2];

  /// The transformer.
  Transformer transformer;

  /// Transformed fragment
  TransformedFragment transformed[2];

  /// Counts the number of stages
  int stage_index;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE Volta884ComplexSharedLoadStream() : stage_index(0) {}

  /// Ctor.
  CUTLASS_DEVICE Volta884ComplexSharedLoadStream(Params const &_params,
                                                 TensorRef const &ref)
      : iterator(ref), stage_index(0) {}

  /// Load the data from shared memory to the fetch fragment.
  CUTLASS_DEVICE void copy(int step) {
    iterator.load(fetched[step % 2],
                  make_Coord(step + stage_index * Iterator::First::VectorizedShape::kD, 0, 0, 0));
  }

  /// Commit the data.
  CUTLASS_DEVICE void commit(int step) {
    transformer.transform(fetched[step % 2], transformed[step % 2]);
  }

  /// Gets the transformed fragment
  CUTLASS_DEVICE
  TransformedFragment &fragment(int step) { return transformed[step % 2]; }

  /// Gets the transformed fragment
  CUTLASS_DEVICE
  TransformedFragment const &fragment(int step) const { return transformed[step % 2]; }

  /// Increment the stage.
  CUTLASS_DEVICE void inc_stage() {
    ++stage_index;
    if (kStageCount && stage_index == StageCount) {
      stage_index = 0;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
