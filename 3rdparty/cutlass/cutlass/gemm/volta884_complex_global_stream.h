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
   storing
      to shared memory.
*/

#pragma once

// clang-format off

#include "cutlass/convert.h"
#include "cutlass/zip_tile_iterator.h"
#include "cutlass/zip_tensor_ref.h"
#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/util/pair.h"

#include "cutlass/gemm/mma_global_stream.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

///! Stream adapter for loading threadblock-scoped GEMM tiles and storing to shared memory
template <
    /// Identifies multiplicand
    GemmOperand::Kind Operand,
    /// Layout of source matrix in global memory
    MatrixLayout::Kind Layout,
    /// Iterator for loading threadblock-scoped tiles
    typename LoadIterator_,
    /// Transformation functor for transforming fragments
    typename Transformer_,
    /// Iterator for storing threadblock-scoped tiles to shared memory
    typename StoreIterator_,
    /// Number of stores before iterator wraps - zero indicates no wrapping
    int StageCount>
struct Volta884ComplexGlobalLoadStream {

  //
  // Type definitions
  //

  /// Identifies the operand
  static GemmOperand::Kind const kOperand = Operand;

  /// The layout.
  static MatrixLayout::Kind const kLayout = Layout;

  /// Load-store stream for real-valued matrices
  typedef MMAGlobalLoadStream<Operand, Layout, LoadIterator_, Transformer_, StoreIterator_, StageCount> RealLoadStoreStream;

  /// Loads a pair of real-valued fragments
  typedef ZipTileIterator<LoadIterator_, LoadIterator_> LoadIterator;

  /// Zips a pair of transformers
  typedef ZipConvert<Transformer_, Transformer_> Transformer;

  /// Stores a pair of real-valued ragments
  typedef ZipTileIterator<StoreIterator_, StoreIterator_> StoreIterator;

  /// Number of stages
  static int const kStageCount = StageCount;

  /// Predicate vector
  typedef typename RealLoadStoreStream::PredicateVector PredicateVector;

  /// The fragment that is copied from shared memory.
  typedef typename LoadIterator::Fragment FetchedFragment;
  /// The fragment that is obtained after the transformation by the transformer.
  typedef typename Transformer::OutputFragment TransformedFragment;
  /// Make sure the fragments match.
  static_assert((platform::is_same<FetchedFragment, typename Transformer::InputFragment>::value),
                "");
  /// The output fragment.
  typedef TransformedFragment Fragment;
  /// Make sure the transformed fragment is the same as the store fragment.
  static_assert((platform::is_same<TransformedFragment, typename StoreIterator::Fragment>::value),
                "");

  /// Index type
  typedef typename RealLoadStoreStream::Index Index;
  
  /// Long index type
  typedef typename RealLoadStoreStream::LongIndex LongIndex;

  /// The params.
  struct Params {

    //
    // Type definitions
    //

    /// Matrix reference
    typedef ZipTensorRef<
      TensorRefBatchStrided<half const, 2>,
      TensorRefBatchStrided<half const, 2> > SourceTensorRef;

    /// Helper
    static int const kElementsPerLdg = LoadIterator::First::Tile::kC;

    //
    // Data members
    //

    /// Source tensor reference
    platform::Pair<LongIndex, LongIndex> batch_stride;

    // The load iterator.
    typename LoadIterator::Params load_iterator;

    // Offset to residue.
    Index offset_to_residue;
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() {}

    ///
    CUTLASS_HOST_DEVICE
    Params(SourceTensorRef const &ref, Index _offset_to_residue) {
      initialize(ref, _offset_to_residue);
    }

    CUTLASS_HOST_DEVICE
    int initialize(SourceTensorRef const &ref, Index _offset_to_residue) {

      batch_stride.first = ref.first.tensor_stride;
      batch_stride.second = ref.second.tensor_stride;

      offset_to_residue = _offset_to_residue;
      load_iterator.first.initialize(
        TensorRef<half const, 4>(
          ref.first.at().data(),
          make_Coord(ref.first.at().stride(0) * kElementsPerLdg, ref.first.at().stride(0), kElementsPerLdg)
        )
      );
      load_iterator.second.initialize(
        TensorRef<half const, 4>(
          ref.second.at().data(),
          make_Coord(ref.second.at().stride(0) * kElementsPerLdg, ref.second.at().stride(0), kElementsPerLdg)
        )
      );
      return 0;
    }
  };

  /// Empty shared storage
  struct SharedStorage {};

  /// Shared memory allocation for the tile
  typedef TileAllocation<
    typename RealLoadStoreStream::StoreIterator::Scalar,
    typename ShapeMul<
      typename RealLoadStoreStream::StoreIterator::OperandShape,
      Shape<kStageCount, 1, 1, 1>
    >::Shape
  > RealThreadblockTileStorage;

  /// Threadblock tile allocation
  typedef ZipTileAllocation<
    RealThreadblockTileStorage,
    RealThreadblockTileStorage
  > ThreadblockTileStorage;

  /// Reference to ThreadblockTileStorage
  typedef typename ThreadblockTileStorage::TensorRef ThreadblockTileRef;

  //
  // Data members
  //

  ///! The parameters
  Params params;

  ///! Dimensions of global memory tile
  Coord<3> threadblock_offset;

  ///! Multiplicand bounds
  Coord<3> multiplicand_bounds;

  ///! Iterator to load threadblock tiles from global memory
  LoadIterator load_iterator;

  ///! Predicate vector
  PredicateVector predicates;

  ///! The fragment to fetch from shared memory.
  FetchedFragment fetched_fragment;

  ///! Functor to transform fragments after they have been loaded
  Transformer transformer;

  ///! The fragment to convert the data after it has been fetched from shared memory.
  TransformedFragment transformed_fragment;

  ///! Iterator to store threadblock tiles to shared memory
  StoreIterator store_iterator;

  ///! Counter
  int stage_index;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE Volta884ComplexGlobalLoadStream(Params const &_params,
                                          SharedStorage &shared_storage,
                                          ThreadblockTileRef const &threadblock_tile_ref,
                                          Coord<3> const bounds,
                                          Coord<3> const &block)
      : params(_params),
        threadblock_offset(RealLoadStoreStream::project_coordinate(block)),
        multiplicand_bounds(RealLoadStoreStream::project_coordinate(bounds, 1)),
        load_iterator(params.load_iterator, threadblock_offset),
        transformer(),
        store_iterator(threadblock_tile_ref),
        stage_index(0) {

    // initialize predicates used to guard loads
    load_iterator.initialize_predicates(
        predicates.begin(), multiplicand_bounds, threadblock_offset);
  }

  /// Loads the data from global memory
  CUTLASS_DEVICE void copy() {
    load_iterator.load_post_increment(fetched_fragment, predicates.begin());
  }

  /// Transform and commit the data to shared memory
  CUTLASS_DEVICE void commit() {

    transformer.transform(fetched_fragment, transformed_fragment);
    store_iterator.store_post_increment(transformed_fragment);

    ++stage_index;
    if (kStageCount && stage_index == kStageCount) {
      store_iterator -= kStageCount;
      stage_index = 0;
    }
  }

  /// Computes a predicate mask for loads during final threadblock tile load iteration
  CUTLASS_DEVICE void residue(Index k, bool skip_clear = false) {
    // That's the residue!
    Coord<3> _block_offset = threadblock_offset;
    if (kOperand == GemmOperand::kA ^ kLayout == MatrixLayout::kRowMajor) {
      // K-strided
      _block_offset =
          make_Coord(threadblock_offset[0], multiplicand_bounds[1] - k, threadblock_offset[2]);
    } else {
      // K-contiguous
      _block_offset = make_Coord(threadblock_offset[0],
                                 threadblock_offset[1],
                                 multiplicand_bounds[2] - k / LoadIterator::First::Tile::kC);
    }

    load_iterator.initialize_predicates(predicates.begin(), multiplicand_bounds, _block_offset);
    fetched_fragment.clear();
  }

  CUTLASS_DEVICE void move_to_residue(Index k, Index kTileK) {}

  CUTLASS_DEVICE void rollback() {}

  /// Adds a Coord<3> to the underlying global load iterator
  CUTLASS_DEVICE Volta884ComplexGlobalLoadStream &operator+=(Coord<3> const &offset) {
    load_iterator += offset;
    return *this;
  }

  /// Adds an offset based on batch stride
  CUTLASS_DEVICE Volta884ComplexGlobalLoadStream &add_batch_offset(int batch_id) {
    load_iterator.first.add_pointer_offset(params.batch_stride.first * batch_id);
    load_iterator.second.add_pointer_offset(params.batch_stride.second * batch_id);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

// clang-format on
