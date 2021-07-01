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

// clang-format off

#include "cutlass/convert.h"
#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tile_allocation.h"

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
struct MMAGlobalLoadStream {
  //
  // Type definitions
  //

  /// Identifies the operand
  static GemmOperand::Kind const kOperand = Operand;
  /// The layout.
  static MatrixLayout::Kind const kLayout = Layout;
  /// The load iterator.
  typedef LoadIterator_ LoadIterator;
  /// The transformer.
  typedef Transformer_ Transformer;
  /// The store iterator to write to shared memory.
  typedef StoreIterator_ StoreIterator;
  /// Number of stages
  static int const kStageCount = StageCount;

  /// Predicate vector
  typedef typename LoadIterator::PredicateVector PredicateVector;
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

  /// The scalar type of the iterator.
  typedef typename LoadIterator::Scalar Scalar;
  /// The pointer.
  typedef typename LoadIterator::Pointer Pointer;
  /// The index.
  typedef typename LoadIterator::Index Index;
  /// The index.
  typedef typename LoadIterator::LongIndex LongIndex;
  /// The tile.
  typedef typename LoadIterator::Tile Tile;

  /// The params.
  struct Params {

    /// Helper
    static int const kElementsPerLdg = LoadIterator::Tile::kC;

    //
    // Data members
    //

    /// The load iterator.
    typename LoadIterator::Params load_iterator;

    /// Stride within a batch of matrix operands
    LongIndex batch_stride;

    // Offset to residue.
    Index offset_to_residue;

    // Offset to residue for the last partition
    Index offset_to_residue_last_partition;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): batch_stride(0), offset_to_residue(0), offset_to_residue_last_partition(0) {}

    /// Constructor
    CUTLASS_HOST_DEVICE
    Params(
      TensorRef<half const, 2> const &ref,
      Index _offset_to_residue
    ):
      batch_stride(0),
      offset_to_residue(_offset_to_residue),
      offset_to_residue_last_partition(0),
      load_iterator(
        TensorRef<half const, 4>(
          ref.data(),
          make_Coord(ref.stride(0) * kElementsPerLdg, ref.stride(0), kElementsPerLdg, 1)
        )
      ) {}

    /// Initializer
    CUTLASS_HOST_DEVICE
    int initialize(
        TensorRef<half const, 2> const &ref,
        LongIndex batch_stride_,
        Index offset_to_residue_,
        Index offset_to_residue_last_partition_) {

      batch_stride = batch_stride_;
      offset_to_residue = offset_to_residue_;
      offset_to_residue_last_partition = offset_to_residue_last_partition_;

      return load_iterator.initialize(
        TensorRef<half const, 4>(
          ref.data(),
          make_Coord(static_cast<int>(batch_stride), ref.stride(0), kElementsPerLdg, 1)
        )
      );
    }

    CUTLASS_HOST_DEVICE
    int initialize(
        TensorRef<half const, 2> const &ref,
        Index offset_to_residue_) {

      offset_to_residue = offset_to_residue_;
      return load_iterator.initialize(
        TensorRef<half const, 4>(
          ref.data(),
          make_Coord(ref.stride(0) * kElementsPerLdg, ref.stride(0), kElementsPerLdg, 1)
          )
      );
    }

    CUTLASS_HOST_DEVICE int initialize(Index offset_to_residue_) {
      offset_to_residue = offset_to_residue_;
      return 0;
    }

    CUTLASS_DEVICE Index get_offset_to_residue() {
      if (blockIdx.z == gridDim.z - 1) { //last partition
        return offset_to_residue_last_partition;
      }
      else {
        return offset_to_residue;
      }
    }
  };

  /// Empty shared storage
  struct SharedStorage {};

  /// Shared memory allocation for the tile
  typedef TileAllocation<
    typename StoreIterator::Scalar,
    typename ShapeMul<
      typename StoreIterator::OperandShape,
      Shape<kStageCount, 1, 1, 1>
    >::Shape
  > ThreadblockTileStorage;

  /// ZipTensorRef to threadblock tiles
  typedef typename ThreadblockTileStorage::TensorRef ThreadblockTileRef;

  //
  // Data members
  //

  ///! The parameters
  Params params;

  ///! Dimensions of global memory tile
  Coord<3> threadblock_offset;

  ///! Dimensions of multiplicand bounds
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
  // Static member functions
  //

  /// Maps a coordinate in the GEMM's (K, N, M) coordinate system to global memory
  CUTLASS_HOST_DEVICE
  static Coord<3> project_coordinate(Coord<3> const &coord, Index d_offset = 0) {
    bool const kKstrided =
        gemm::GemmMultiplicandTraits<typename LoadIterator::Tile, kOperand, kLayout>::kKstrided;

    Coord<3> tile_coord = gemm::ProjectOperand<kOperand, kKstrided>::project(coord);

    return make_Coord(
        tile_coord[0] + d_offset, tile_coord[1], tile_coord[2] / LoadIterator::Tile::kC);
  }

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE MMAGlobalLoadStream(Params const &_params,
                                      SharedStorage &shared_storage,
                                      ThreadblockTileRef const &threadblock_tile_ref,
                                      Coord<3> const bounds,
                                      Coord<3> const &block)
      : params(_params),
        threadblock_offset(project_coordinate(block)),
        multiplicand_bounds(project_coordinate(bounds, 1)),
        load_iterator(params.load_iterator, threadblock_offset),
        transformer(),
        store_iterator(threadblock_tile_ref.data()),
        stage_index(0) {
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
                                 multiplicand_bounds[2] - k / LoadIterator::Tile::kC);
    }

    load_iterator.initialize_predicates(predicates.begin(), multiplicand_bounds, _block_offset);
    fetched_fragment.clear();
  }

  /// Move to the residue portion.
  CUTLASS_DEVICE void move_to_residue(Index k, Index kTileK) {
    Index kResidue = k % kTileK;
    if (kResidue) {
      residue(kResidue);
      Index this_offset_residue = params.get_offset_to_residue();
      load_iterator.add_pointer_offset(this_offset_residue * load_iterator.stride_advance());
    }
  }

  /// Rollback to the beginning of the first tile
  CUTLASS_DEVICE void rollback(void) {
    load_iterator.initialize_predicates(predicates.begin(), multiplicand_bounds, threadblock_offset);

    int const kBlock = kOperand == GemmOperand::kA
                           ? (kLayout == MatrixLayout::kColumnMajor ? Tile::kH : Tile::kW)
                           : (kLayout == MatrixLayout::kRowMajor ? Tile::kH : Tile::kW);
    Index this_offset_residue = params.get_offset_to_residue();
    load_iterator.add_pointer_offset(-(this_offset_residue + kBlock) *
                                     load_iterator.stride_advance());
  }

  /// Adds a Coord<3> to the underlying global load iterator
  CUTLASS_DEVICE MMAGlobalLoadStream &operator+=(Coord<3> const &offset) {
    load_iterator += offset;
    return *this;
  }

  /// Adds an offset based on batch stride
  CUTLASS_DEVICE MMAGlobalLoadStream &add_batch_offset(int batch_id) {
    load_iterator.add_pointer_offset(batch_id * params.batch_stride);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // gemm 
}  // namespace cutlass

// clang-format on
