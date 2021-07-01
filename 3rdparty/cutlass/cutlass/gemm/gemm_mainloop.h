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
    \brief Implements a software-pipelined efficient GEMM.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace cutlass {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits_>
struct GemmMainloop {

  //
  // Type definitions
  //

  /// The traits.
  typedef Traits_ Traits;

  /// The GEMM mainloop
  typedef typename Traits::KernelClass KernelClass;

  /// The shared storage.
  typedef typename Traits::SharedStorage SharedStorage;

  /// The scalar for A.
  typedef typename Traits::ScalarA ScalarA;
  /// The scalar for B.
  typedef typename Traits::ScalarB ScalarB;
  /// The scalar in the epilogue.
  typedef typename Traits::Epilogue::Scalar ScalarEpilogue;
  /// The scalar for C.
  typedef typename Traits::Epilogue::ScalarC ScalarC;
  /// The scalar for D.
  typedef typename Traits::Epilogue::ScalarD ScalarD;
  /// The index.
  typedef typename Traits::Index Index;

  /// Define the mainloop iteration size
  typedef typename Traits::MultiplyAdd MultiplyAdd;

  /// The number of threads.
  static int const kThreads = Traits::GemmConfig::kThreads;

  // Number of warp-level multiply-accumulate steps executed by each warp.
  static Index const kWarpGemmSteps =
      Traits::GemmConfig::AccumulatorsPerWarp::kD / MultiplyAdd::InstructionShape::kD;

  /*
  // Make sure we have at least 2 unrolling steps or our pipeling is not going to work.
  static_assert(kWarpGemmSteps >= 2, "The pipelining assumes at least two steps");
  */

  /// Use the params object defined in traits
  typedef typename Traits::Params Params;

  //
  // Data members
  //

  /// The params.
  Params const& params;

  /// SharedStorage object
  SharedStorage& shared_storage;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE GemmMainloop(Params const& params_, SharedStorage& shared_storage_)
      : params(params_), shared_storage(shared_storage_) {}

  /// Fetches global stream pair
  template <bool Residue>
  CUTLASS_DEVICE void fetch_global(typename Traits::GlobalLoadStream& global_to_shared_stream,
                                   Index outer_k) {
    // If residue portion and not calculating residue in prolog, update residue predicates now.
    if (Residue) {
      global_to_shared_stream.residue(outer_k);
    }
    global_to_shared_stream.copy();
  }

  /// Computes a warp-level GEMM on data held in shared memory
  template <bool Residue, bool LastIteration>
  CUTLASS_DEVICE void consume_tile(typename Traits::GlobalLoadStream& global_to_shared_stream,
                                   typename Traits::SharedStream& shared_load_stream,
                                   typename MultiplyAdd::Accumulators& accumulators,
                                   Index outer_k) {

    // Whether to load global stream before loading shared stream
    const bool kGlobalStreamFirst = (kWarpGemmSteps <= 4);

    // Load data for the next iteration of the main loop (unless it's the last iteration).
    if (kGlobalStreamFirst && !LastIteration) {
      fetch_global<Residue>(global_to_shared_stream, outer_k);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int step = 0; step < kWarpGemmSteps; ++step) {

      // Trigger the copy from shared memory for the next A/B values.
      shared_load_stream.copy((step + 1) % kWarpGemmSteps);

      // Load data for the next iteration of the main loop (unless it's the last iteration).
      if (!kGlobalStreamFirst && (step == 0) && !LastIteration) {
        fetch_global<Residue>(global_to_shared_stream, outer_k);
      }

      if (step == kWarpGemmSteps - 2) {
          // Make sure the data from shared memory has been entirely consumed.
          Traits::shared_load_fence(true);

          global_to_shared_stream.commit();

          // Make sure the data is in shared memory.
          Traits::shared_store_fence(true);

          // Move to the next stage for the load (if it makes sense).
          shared_load_stream.inc_stage();
      }

      // Make sure the values are available for the current iteration to do the multiply-add.
      shared_load_stream.commit(step);

      // Do the math on the fragments of the current iteration.
      MultiplyAdd multiply_add;
      multiply_add.multiply_add(shared_load_stream.fragment_a(step),
                                shared_load_stream.fragment_b(step),
                                accumulators,
                                accumulators);
    }
  }

  /// Do the GEMM.
  CUTLASS_DEVICE void multiply_add() {
    // Swizzle the IDs of the block (to enable better cache behavior).
    typename Traits::BlockSwizzle block_swizzle;
    Coord<3> threadblock_offset =
        block_swizzle.get_threadblock_offset(make_Coord_from_shape<typename Traits::OutputTile>());

    // We may want to use shared memory to clear the registers.
    typedef typename Traits::ClearAccumulators ClearAccumulators;

    // Get the bounds for each thread, it maybe different than problem_size
    Coord<3> bounds = block_swizzle.get_threadblock_bounds(params.problem_size,
                                                        params.partitionK_range);

    // The streams to read A/B from global memory to shared memory.
    typename Traits::GlobalLoadStream global_to_shared_stream(
        params.global_to_shared_stream,
        shared_storage.main_loop.global_to_shared_stream,
        shared_storage.main_loop.threadblock_tile.reference(),
        bounds,
        threadblock_offset);

    // update A and B pointer offset based on batch_id and batch_stride_offset
    global_to_shared_stream.add_batch_offset(block_swizzle.get_batch_id());

    // Create the accumulator clear.
    ClearAccumulators clear;

    // Deal with residue in prolog.
    // global_to_shared_stream.move_to_residue(params.problem_size[0], Traits::OutputTile::kD);
    global_to_shared_stream.move_to_residue(bounds[0], Traits::OutputTile::kD);

    // Fetch the fragments for A and B from global memory.
    global_to_shared_stream.copy();

    // Copy the elements to shared memory (after transformation if needed).
    global_to_shared_stream.commit();

    // Make sure the data is in shared memory.
    Traits::shared_store_fence(false);

    // Rollback to the beginning of the first tile (if residue exists).
    // global_to_shared_stream.rollback(params.problem_size[0] % Traits::OutputTile::kD);
    global_to_shared_stream.rollback(bounds[0] % Traits::OutputTile::kD);

    // The stream of data from shared memory to fragments.
    typename Traits::SharedStream shared_load_stream(
        params.shared_stream,
        shared_storage.main_loop.threadblock_tile.reference());

    // Trigger the copy from shared memory for the 1st stream.
    shared_load_stream.copy(0);

    // Allocate the accumulators.
    typename MultiplyAdd::Accumulators accumulators;

    // Clear the accumulators.
    clear.clear(accumulators);

    // Initial index
    // Index outer_k = params.problem_size[0] - Traits::OutputTile::kD;
    // problem_size[0] might be bigger than bounds[0]
    Index outer_k = bounds[0] - Traits::OutputTile::kD;
    // Check if we are computing residue in prolog or not.
    if (Traits::GemmConfig::kResidueInProlog) {
      // Execute all mainloop iterations but the last one.

      CUTLASS_GEMM_LOOP
      for (; outer_k > 0; outer_k -= Traits::OutputTile::kD) {
        CUTLASS_GEMM_LOOP_HEADER
        consume_tile<false, false>(
            global_to_shared_stream, shared_load_stream, accumulators, outer_k);
      }

      consume_tile<false, true>(
          global_to_shared_stream, shared_load_stream, accumulators, outer_k);

    } else {
      // When kResidueSeparate = true, execute all mainloop iterations but the last two without any
      // consideration for K-residue or predicate updates. This improves the steady state of some
      // kernels.
      if (Traits::GemmConfig::kResidueSeparate) {

        CUTLASS_GEMM_LOOP
        for (; outer_k > Traits::OutputTile::kD; outer_k -= Traits::OutputTile::kD) {
          CUTLASS_GEMM_LOOP_HEADER
          consume_tile<false, false>(
              global_to_shared_stream, shared_load_stream, accumulators, outer_k);
        }
      }

      // Execute remaining tiles with K-residue predicate updates enabled.
      CUTLASS_GEMM_LOOP
      for (; outer_k > -Traits::OutputTile::kD; outer_k -= Traits::OutputTile::kD) {
        CUTLASS_GEMM_LOOP_HEADER
        consume_tile<true, false>(
            global_to_shared_stream, shared_load_stream, accumulators, outer_k);
      }
    }

    typedef typename Traits::Epilogue Epilogue;
    Epilogue epilogue(params.epilogue, shared_storage.epilogue, params.problem_size.knm());
    epilogue.epilogue(accumulators, threadblock_offset, block_swizzle.get_batch_id());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass
