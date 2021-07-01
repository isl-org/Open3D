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
    \brief Defines a pair of GEMM tile streams
*/
#pragma once

#include "cutlass/convert.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_allocation.h"
#include "cutlass/tile_iterator.h"

#include "cutlass/gemm/clear_accumulators.h"
#include "cutlass/gemm/gemm_config.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/threadblock_swizzle.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Collect the global load streams for multiplicands.
template <typename StreamA_, typename StreamB_, bool kResidueInProlog_>
struct GlobalLoadStreamPair {
  //
  // Type definitions
  //

  /// Stream for A multiplicand
  typedef StreamA_ StreamA;

  /// Stream for B multiplicand
  typedef StreamB_ StreamB;

  /// Parameters object
  struct Params {
    /// Parameters object for StreamA
    typename StreamA::Params stream_a;

    /// Parameters object for StreamB
    typename StreamB::Params stream_b;

    /// Default constructor
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Constructs a global load stream pair Params object
    CUTLASS_HOST_DEVICE
    Params(typename StreamA::Params const &_params_A, typename StreamB::Params const &_params_B)
        : stream_a(_params_A), stream_b(_params_B) {}
  };

  /// Assumes the A stream defines the index type
  typedef typename StreamA::Index Index;

  /// Shared memory allocation for threadblock-scoped GEMM tile
  typedef ZipTileAllocation<typename StreamA::ThreadblockTileStorage,
                              typename StreamB::ThreadblockTileStorage>
      ThreadblockTileStorage;

  /// ZipTensorRef to threadblock tiles
  typedef typename ThreadblockTileStorage::TensorRef ThreadblockTileRef;

  /// Defines a structure containing shared storage for each pair
  struct SharedStorage {
    typename StreamA::SharedStorage stream_a;
    typename StreamB::SharedStorage stream_b;
  };

  //
  // Data members
  //

  /// Stream for A multiplicand
  StreamA stream_a;

  /// Stream for B multiplicand
  StreamB stream_b;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE GlobalLoadStreamPair(Params const &params,
                                      SharedStorage &shared_storage,
                                      ThreadblockTileRef const &threadblock_tile_ref,
                                      Coord<3> const bounds,
                                      Coord<3> const &block_offset = make_Coord(0, 0, 0))
      : stream_a(params.stream_a,
                 shared_storage.stream_a,
                 threadblock_tile_ref.first,
                 bounds,
                 block_offset),
        stream_b(params.stream_b,
                 shared_storage.stream_b,
                 threadblock_tile_ref.second,
                 bounds,
                 block_offset) {}

  CUTLASS_DEVICE
  GlobalLoadStreamPair & operator+=(Coord<3> const offset) {
    stream_a += offset;
    stream_b += offset;
    return *this;
  }

  CUTLASS_DEVICE
  GlobalLoadStreamPair & add_batch_offset(int batch_id) {
    stream_a.add_batch_offset(batch_id);
    stream_b.add_batch_offset(batch_id);
    return *this;
  }

  /// Trigger the copies from shared memory to registers.
  CUTLASS_DEVICE void copy() {

    stream_a.copy();

    stream_b.copy();

  }

  /// Commit the data.
  CUTLASS_DEVICE void commit() {
    stream_a.commit();

    stream_b.commit();

  }

  /// Execute the residue code.
  CUTLASS_DEVICE void residue(Index k, bool skip_clear = false) {
    stream_a.residue(k, skip_clear);
    stream_b.residue(k, skip_clear);
  }

  /// Move to residue.
  CUTLASS_DEVICE void move_to_residue(Index k, Index kTileK) {
    if (kResidueInProlog_) {
      stream_a.move_to_residue(k, kTileK);
      stream_b.move_to_residue(k, kTileK);
    } else if (k < kTileK) {
      residue(k, true);
    }
  }

  /// Rollback to beginning of first tile.
  CUTLASS_DEVICE void rollback(bool kRollback) {
    if (kResidueInProlog_ && kRollback) {
      stream_a.rollback();
      stream_b.rollback();
    }
  }
};

/// Collect the global load streams for multiplicands.
template <typename StreamA_, typename StreamB_>
struct SharedStreamPair {
  //
  // Type definitions
  //

  /// Stream for A multiplicand
  typedef StreamA_ StreamA;

  /// Stream for B multiplicand
  typedef StreamB_ StreamB;

  /// Parameters object passed to load iterators
  struct Params {
    ///
    typename StreamA::Params stream_a;

    ///
    typename StreamB::Params stream_b;
  };

  /// Shared memory allocation for threadblock-scoped GEMM tile
  typedef ZipTensorRef<typename StreamA::TensorRef,
                       typename StreamB::TensorRef >
      ThreadblockTileRef;

  //
  // Data members
  //

  /// The stream for A.
  StreamA stream_a;

  /// The stream for B.
  StreamB stream_b;

  //
  // Methods
  //

  /// Construct with the composable structure
  CUTLASS_DEVICE SharedStreamPair(Params const &params, ThreadblockTileRef const &threadblock_tile_ref)
      : stream_a(params.stream_a, threadblock_tile_ref.first),
        stream_b(params.stream_b, threadblock_tile_ref.second) {}

  /// Trigger the copies from shared memory to registers.
  CUTLASS_DEVICE void copy(int step) {
    stream_a.copy(step);
    stream_b.copy(step);
  }

  /// Commit the data.
  CUTLASS_DEVICE void commit(int step) {
    stream_a.commit(step);
    stream_b.commit(step);
  }

  /// Clears all fragments
  CUTLASS_DEVICE
  void clear() {
    stream_a.clear();
    stream_b.clear();
  }

  /// The fragment A.
  CUTLASS_DEVICE
  typename StreamA::TransformedFragment const &fragment_a(int step) const {
    return stream_a.fragment(step);
  }

  /// The fragment B.
  CUTLASS_DEVICE
  typename StreamB::TransformedFragment const &fragment_b(int step) const {
    return stream_b.fragment(step);
  }

  /// Increment the stage.
  CUTLASS_DEVICE void inc_stage() {
    stream_a.inc_stage();
    stream_b.inc_stage();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
