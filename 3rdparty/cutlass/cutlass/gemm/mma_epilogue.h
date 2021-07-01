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
    \brief Implements the epilogue phase of the GEMM kernel that efficiently updates global memory
   with
      the computed matrix product.
*/

#pragma once

// clang-format off

#include "cutlass/coord.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename EpilogueTraits_>
struct MMAEpilogue {
  /// The traits class.
  typedef EpilogueTraits_ Traits;

  /// The params.
  typedef typename Traits::Params Params;

  /// The shared storage.
  typedef typename Traits::SharedStorage SharedStorage;

  /// Defines a tiling of the EpilogueTile over the entire threadblock GEMM tile
  typedef typename Traits::Iterations Iterations;

  /// The output tile.
  typedef typename Traits::OutputTile OutputTile;

  /// Accumulators to store in the epilogue
  typedef typename Traits::Accumulators Accumulators;

  /// A functor to copy a slice of accumulators for a given epilogue iteration
  typedef typename Traits::SelectAccumulators SelectAccumulators;

  /// The iterator to load source matrix from global memory.
  typedef typename Traits::GlobalLoadStreamC GlobalLoadStreamC;

  /// The iterator to store the final GEMM computation to global memory.
  typedef typename Traits::GlobalStoreStreamD GlobalStoreStreamD;

  /// The stream to store matrix product to shared memory
  typedef typename Traits::SharedStoreStreamD SharedStoreStreamD;

  /// The stream to load the matrix product from shared memory
  typedef typename Traits::SharedLoadStreamD SharedLoadStreamD;

  /// The functor in charge of the math.
  typedef typename Traits::Functor Functor;

  /// The scalar type used by the epilogue functor.
  typedef typename Functor::Scalar Scalar;

  /// The scalar type of the source accumulator matrix.
  typedef typename Traits::ScalarC ScalarC;

  /// The scalar type of the destination accumulator matrix.
  typedef typename Traits::ScalarD ScalarD;

  /// The index type.
  typedef typename Traits::Index Index;

  /// Functor computing the offset from the threadblock origin per iteration of
  /// the epilogue.
  typedef typename Traits::GlobalOffset GlobalOffset;

  ///
  typedef typename Traits::GlobalDataLayout GlobalDataLayout;

  //
  // Data members
  //

  /// The params.
  Params const& params;

  /// The shared storage.
  SharedStorage& shared_storage;

  /// The dimensions of the GEMM.
  gemm::GemmCoord problem_size;

  /// Epilogue functor
  Functor functor;

  // Functor to select a set of accumulators
  SelectAccumulators select_accumulators;


  // Functor to compute the global offset relative to the threadblock for each iteration
  // of the epilogue.
  GlobalOffset global_offset;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE MMAEpilogue(
      Params const& params_,
      SharedStorage& shared_storage_,
      Coord<3> const& _problem_size,
      SelectAccumulators _select_accumulators = SelectAccumulators(),
      GlobalOffset _global_offset = GlobalOffset()
  ):
    params(params_),
    shared_storage(shared_storage_),
    problem_size(_problem_size),
    functor(params_.functor),
    select_accumulators(_select_accumulators),
    global_offset(_global_offset) {}

  /// Execute the epilogue.
  CUTLASS_DEVICE void epilogue(
      Accumulators& accumulators,
      Coord<3> const& threadblock_offset = make_Coord(0, 0, 0),
      int batch_id = 0) {

    if (functor.source_required()) {
      epilogue_with_or_without_beta<true>(accumulators, threadblock_offset, batch_id);
    }
    else {
      epilogue_with_or_without_beta<false>(accumulators, threadblock_offset, batch_id);
    }
  }

  ///

  /// Execute the epilogue.
  template <bool kSourceRequired>
  CUTLASS_DEVICE void epilogue_with_or_without_beta(
      Accumulators& accumulators,
      Coord<3> const& threadblock_offset = make_Coord(0, 0, 0),
      int batch_id = 0) {

    /// Global memory mapping function
    GlobalDataLayout gmem_map_func;

    // Construct shared memory streams
    SharedStoreStreamD shared_store_stream(
      params.shared_store_stream_d,
      shared_storage.reference());

    SharedLoadStreamD shared_load_stream(
      params.shared_load_stream_d,
      shared_storage.reference());

    // Map the GEMM problem dimensions into the coordinate system of the output memory
    Coord<2> gmem_bounds = gmem_map_func(make_Coord(
      problem_size.m(),   // GEMM M - rows
      problem_size.n())); // GEMM N - columns

    Coord<3> gmem_tile_bounds = make_Coord(
      problem_size.k(),   // GEMM K
      gmem_bounds[0],     // strided
      gmem_bounds[1]);    // contiguous

    // Iterate over the entire Threadblock tile
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < Iterations::kH; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < Iterations::kW; ++w) {
        if (!(h == 0)) {
          //continue;
        }

        // Offset in GEMM coordinates
        gemm::GemmCoord offset_in_gemm = threadblock_offset + global_offset(make_Coord(h, w));

        Coord<2> offset_in_memory = gmem_map_func(
          make_Coord(
            offset_in_gemm.m(),       // GEMM M - rows
            offset_in_gemm.n()));     // GEMM N - columns

        // Offset in
        Coord<3> global_tile_offset = make_Coord(
          offset_in_gemm.k(),         // GEMM K
          offset_in_memory[0],        // strided
          offset_in_memory[1]);       // contiguous

        GlobalLoadStreamC global_load_stream(
          params.load_stream_c,
          gmem_tile_bounds,
          global_tile_offset);

        GlobalStoreStreamD global_store_stream(
          params.store_stream_d,
          gmem_tile_bounds,
          global_tile_offset);

        // update C pointer offset based on batch_id and batch_stride_offset
        global_load_stream.iterator.add_pointer_offset(batch_id * params.batch_stride_C);

        // update D pointer offset based on batch_id and batch_stride_offset
        global_store_stream.iterator.add_pointer_offset(batch_id * params.batch_stride_D);

        // Load the C matrix into fragment.
        if (kSourceRequired) {
          global_load_stream.copy();
        }

        // Make sure we can write to shared memory.
        shared_load_fence();

        // Store accumulator tile to shared memory
        shared_store_stream.copy(
          select_accumulators(accumulators, make_Coord(h, w)));

        shared_store_stream.commit();

        // Make sure the data is in shared memory.
        shared_store_fence();

        // Load the accumulators back to registers from shared memory.
        shared_load_stream.copy();
        shared_load_stream.commit();
        // Commit the C matrix fragment
        if (kSourceRequired) {
          global_load_stream.commit();
        }

        // Apply epilogue functor
        if (kSourceRequired) {

          functor.evaluate(shared_load_stream.fragment(),
                           global_load_stream.fragment(),
                           global_store_stream.fragment());
        }
        else {

          functor.evaluate(
            shared_load_stream.fragment(),
            global_store_stream.fragment());
        }

        global_store_stream.copy();
        global_store_stream.commit();
      }
    }
  }

  /// The memory fence for shared loads.
  CUTLASS_DEVICE void shared_load_fence() { __syncthreads(); }

  /// The memory fence for shared stores.
  CUTLASS_DEVICE void shared_store_fence() { __syncthreads(); }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // gemm 
}  // namespace cutlass

// clang-format on
