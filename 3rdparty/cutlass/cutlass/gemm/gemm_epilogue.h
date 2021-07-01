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

#include "cutlass/convert.h"
#include "cutlass/coord.h"
#include "cutlass/fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmEpilogueTraits_>
struct GemmEpilogue {
  /// The traits class.
  typedef GemmEpilogueTraits_ Traits;
  /// The params.
  typedef typename Traits::Params Params;
  /// The shared storage.
  typedef typename Traits::SharedStorage SharedStorage;

  /// The output tile.
  typedef typename Traits::OutputTile OutputTile;
  /// The number of iterations.
  typedef typename Traits::Iterations Iterations;
  /// The accumulators.
  typedef typename Traits::Accumulators Accumulators;
  /// The scalar.
  typedef typename Traits::Scalar Scalar;
  /// The functor in charge of the math.
  typedef typename Traits::Functor Functor;

  /// We do not support 3D or 4D shapes.
  static_assert(Iterations::kD == 1 && Iterations::kC == 1, "Unsupported 3D/4D shapes");

  /// The iterator for C in global memory.
  typedef typename Traits::GlobalLoadIteratorC GlobalLoadIteratorC;
  /// The transformer for C.
  typedef typename Traits::GlobalTransformerC GlobalTransformerC;
  /// The transformer for D.
  typedef typename Traits::GlobalTransformerD GlobalTransformerD;
  /// The iterator for D in global memory.
  typedef typename Traits::GlobalStoreIteratorD GlobalStoreIteratorD;
  /// The iterator to store D in shared memory.
  typedef typename Traits::SharedStoreIteratorD SharedStoreIteratorD;
  /// The shared store transformer for D.
  typedef typename Traits::SharedStoreTransformerD SharedStoreTransformerD;
  /// The iterator to load D in shared memory.
  typedef typename Traits::SharedLoadStreamD SharedLoadStreamD;

  /// The index.
  typedef typename Traits::Index Index;

  /// The scalar for C.
  typedef typename GlobalLoadIteratorC::Scalar ScalarC;
  /// The scalar for D.
  typedef typename GlobalStoreIteratorD::Scalar ScalarD;

  /// Ctor.
  CUTLASS_DEVICE GemmEpilogue(Params const& params_,
                              SharedStorage& shared_storage_,
                              Coord<3> const& _problem_size)
      : params(params_), shared_storage(shared_storage_), problem_size(_problem_size), functor(params_.functor) {}

  /// Execute the epilogue.
  CUTLASS_DEVICE void epilogue(Accumulators& accumulators,
                               Coord<3> const& block = make_Coord(0, 0, 0),
                               int batch_id = 0) {
    if (functor.source_required()) {
      epilogue_with_or_without_beta<true>(accumulators, block, batch_id);
    } else {
      epilogue_with_or_without_beta<false>(accumulators, block, batch_id);
    }
  }

  template <bool kSourceRequired>
  CUTLASS_DEVICE void epilogue_with_or_without_beta(Accumulators& accumulators,
                                                    Coord<3> const& block,
                                                    int batch_id) {
    // The C fragment.
    typename GlobalLoadIteratorC::Fragment fragment_c;
    // The transformed C fragment.
    typename GlobalTransformerC::OutputFragment transformed_c;
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < Iterations::kH; ++h) {
      // Compute pointer and predicate offsets for C and D global iterators.
      int const pointer_offset =
          ((params.iterator_d.inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
            params.iterator_d.inc_advance) *
               Iterations::kW +
           params.stride_h) *
          h;

      int const predicate_offset =
          ((params.iterator_d.predicate_inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
            params.iterator_d.predicate_inc_advance) *
               Iterations::kW +
           Traits::Delta::kH) *
          h;

      // The iterator to load the elements of the C matrix.
      GlobalLoadIteratorC global_load_iterator(
          params.iterator_c, problem_size, block, pointer_offset, predicate_offset);

      // update C pointer offset based on batch_id and batch_stride_offset
      global_load_iterator.add_pointer_offset(batch_id * params.batch_stride_C);

      // The transformer for C.
      GlobalTransformerC transformer_c;
      // The transformer for D.
      GlobalTransformerD transformer_d;

      // The iterator to store into the D matrix.
      GlobalStoreIteratorD global_store_iterator(
          params.iterator_d, problem_size, block, pointer_offset, predicate_offset);

      // update D pointer offset based on batch_id and batch_stride_offset
      global_store_iterator.add_pointer_offset(batch_id * params.batch_stride_D);

      SharedStoreTransformerD shared_store_transformer;
      typename SharedStoreTransformerD::OutputFragment shared_store_transformed_d;

      SharedStoreIteratorD shared_store_iterator(
          params.shared_store_iterator_d,
          reinterpret_cast<typename SharedStoreIteratorD::Scalar*>(shared_storage.data()));

      SharedLoadStreamD shared_load_stream(
          params.shared_load_stream_d,
          reinterpret_cast<typename SharedLoadStreamD::Scalar*>(shared_storage.data()));

      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < Iterations::kW; ++w) {
        // Load the C matrix into fragment.
        if (kSourceRequired) {
          global_load_iterator.load_post_increment(fragment_c);
        }

        // Make sure we can write to shared memory.
        shared_load_fence();

        // Copy the accumulators to shared memory.
        int const offset = (h * Iterations::kW + w) * SharedStoreIteratorD::Fragment::kElements;

        shared_store_transformer.transform(accumulators, offset, shared_store_transformed_d);

        shared_store_iterator.store_post_increment(shared_store_transformed_d);

        // Make sure the data is in shared memory.
        shared_store_fence();

        // Copy the accumulators back to registers from shared memory.
        shared_load_stream.copy();
        shared_load_stream.commit();

        // Do the math.
        typename GlobalTransformerD::InputFragment fragment_d;
        if (kSourceRequired) {
          // Transform C fragment.
          transformer_c.transform(fragment_c, transformed_c);
          // Do the math.
          functor.evaluate(shared_load_stream.fragment(), transformed_c, fragment_d);
        } else {
          functor.evaluate(shared_load_stream.fragment(), fragment_d);
        }

        // Transform D fragment.
        typename GlobalTransformerD::OutputFragment global_transformed_d;
        transformer_d.transform(fragment_d, global_transformed_d);

        // Copy the results to global memory.
        global_store_iterator.store_post_increment(global_transformed_d);
      }
    }
  }

  /// The memory fence for shared loads.
  CUTLASS_DEVICE void shared_load_fence() { __syncthreads(); }

  /// The memory fence for shared stores.
  CUTLASS_DEVICE void shared_store_fence() { __syncthreads(); }

  /// The params.
  Params const& params;
  /// The shared storage.
  SharedStorage& shared_storage;
  /// The dimensions of the GEMM.
  Coord<3> problem_size;
  // The functor.
  Functor functor;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
