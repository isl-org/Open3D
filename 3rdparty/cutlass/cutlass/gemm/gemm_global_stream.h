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

#include "cutlass/coord.h"
#include "cutlass/convert.h"
#include "cutlass/gemm/gemm_global_tile.h"
#include "cutlass/tile_allocation.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Identifies multiplicand
    GemmOperand::Kind Operand,
    /// The load iterator.
    typename LoadIterator_,
    /// The store iterator to copy to shared memory.
    typename StoreIterator_,
    /// The transformer to be applied after the data has been copied from global memory.
    typename Transformer_>

struct GlobalLoadStream {
  /// Indicates the type of GEMM operand
  static GemmOperand::Kind const kOperand = Operand;
  /// The load iterator.
  typedef LoadIterator_ LoadIterator;
  /// The transformer.
  typedef Transformer_ Transformer;
  /// The store iterator to write to shared memory.
  typedef StoreIterator_ StoreIterator;

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

  /// The layout.
  static MatrixLayout::Kind const kLayout = LoadIterator::kLayout;
  /// The scalar type of the iterator.
  typedef typename LoadIterator::Scalar Scalar;
  /// The pointer.
  typedef typename LoadIterator::Pointer Pointer;
  /// The index.
  typedef typename LoadIterator::Index Index;
  /// The index.
  typedef typename LoadIterator::LongIndex LongIndex;
  /// The tile
  typedef typename LoadIterator::Tile Tile;

  /// Shared memory allocation for the tile
  typedef TileAllocation<typename StoreIterator::Scalar, typename StoreIterator::Tile>
      ThreadblockTileStorage;

  /// Tensor reference to threadblock tile
  typedef typename ThreadblockTileStorage::TensorRef ThreadblockTileRef;

  /// The params.
  struct Params {
    // The load iterator.
    typename LoadIterator::Params load_iterator;

    /// Batch stride in global memory
    LongIndex batch_stride;

    // The store iterator.
    typename StoreIterator::Params store_iterator;

    // Offset to residue.
    Index offset_to_residue;

    // Offset to residue for the last partition
    Index offset_to_residue_last_partition;

    /// Setup the params.
    CUTLASS_HOST_DEVICE int initialize(Pointer pointer,
                                       LongIndex batch_stride_,
                                       Index ldm,
                                       Index offset_to_residue_,
                                       Index offset_to_residue_last_partition_) {

      int error_code = load_iterator.initialize(pointer, ldm, ldm);
      if (error_code) {
        return error_code;
      }

      batch_stride = batch_stride_;
      offset_to_residue = offset_to_residue_;
      offset_to_residue_last_partition = offset_to_residue_last_partition_;

      return store_iterator.initialize();
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

  /// Contains private storage in shared memory needed by the objects within this class. Note,
  /// this is *NOT* the shared memory allocation for the GEMM threadblock tile. That necessarily
  /// exists outside this class, as it is also needed by the warp-level shared=>RF stream.
  struct SharedStorage {};

  //
  // Static member functions
  //

  /// Maps a coordinate in the GEMM's (K, N, M) coordinate system to global memory
  CUTLASS_HOST_DEVICE static Coord<3> project_coordinate(Coord<3> const& coord, Index d_offset = 0) {
    bool const kKstrided =
        GemmMultiplicandTraits<typename LoadIterator::Tile, kOperand, kLayout>::kKstrided;
    Coord<3> tile_coord = ProjectOperand<kOperand, kKstrided>::project(coord);
    return make_Coord(
        tile_coord[0] + d_offset, tile_coord[1], tile_coord[2] / LoadIterator::Tile::kC);
  }

  /// Ctor.
  CUTLASS_DEVICE GlobalLoadStream(
      Params const& _params,
      SharedStorage& shared_storage,
      ThreadblockTileRef const& threadblock_tile_ref,
      Coord<3> const bounds,
      Coord<3> const& _threadblock_offset)
      : params(_params),
        threadblock_offset(project_coordinate(_threadblock_offset)),
        multiplicand_bounds(project_coordinate(bounds, 1)),
        load_iterator(params.load_iterator, threadblock_offset),
        transformer(),
        store_iterator(params.store_iterator, threadblock_tile_ref.data()) {
    load_iterator.initialize_predicates(multiplicand_bounds, threadblock_offset);
    fetched_fragment.clear();
  }


  /// Load the data from shared memory to the fetch fragment.
  CUTLASS_DEVICE void copy() {
    load_iterator.load_post_increment(fetched_fragment);
  }

  /// Commit the data.
  CUTLASS_DEVICE void commit() {
    transformer.transform(fetched_fragment, transformed_fragment);
    store_iterator.store_post_increment(transformed_fragment);
    store_iterator.inc_stage();
  }

  /// Execute the residue code.
  CUTLASS_DEVICE void residue(Index k, bool skip_clear = false) {
    load_iterator.residue(k);
    if (!skip_clear) {
      fetched_fragment.clear();
    }
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
    load_iterator.initialize_predicates(multiplicand_bounds, threadblock_offset);

    int const kBlock = kOperand == GemmOperand::kA
                           ? (kLayout == MatrixLayout::kColumnMajor ? Tile::kH : Tile::kW)
                           : (kLayout == MatrixLayout::kRowMajor ? Tile::kH : Tile::kW);
    Index this_offset_residue = params.get_offset_to_residue();
    load_iterator.add_pointer_offset(-(this_offset_residue + kBlock) *
      load_iterator.stride_advance());
  }

  /// Adds a Coord<3> to the underlying global load iterator
  CUTLASS_DEVICE GlobalLoadStream &operator+=(Coord<3> const &offset) {
    load_iterator += offset;
    return *this;
  }

  /// Adds an offset based on batch stride
  CUTLASS_DEVICE GlobalLoadStream &add_batch_offset(int batch_id) {
    load_iterator.add_pointer_offset(batch_id * params.batch_stride);
    return *this;
  }

  //
  // Data members
  //

  /// Parameters
  Params params;
  /// Threadblock offset
  Coord<3> threadblock_offset;
  /// Multiplicand bounds
  Coord<3> multiplicand_bounds;
  /// The iterator.
  LoadIterator load_iterator;
  /// The fragment to fetch from shared memory.
  FetchedFragment fetched_fragment;
  /// The transformer.
  Transformer transformer;
  /// The fragment to convert the data after it has been fetched from shared memory.
  TransformedFragment transformed_fragment;
  /// The store iterator.
  StoreIterator store_iterator;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace gemm
}  // namespace cutlass
