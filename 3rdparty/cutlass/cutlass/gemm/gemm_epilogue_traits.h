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
    \brief Defines structural properties of the GEMM epilogue.
*/
#pragma once

#include "cutlass/convert.h"
#include "cutlass/coord.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/linear_scaling.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_iterator.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The output tile.
    typename OutputTile_,
    /// The accumulators.
    typename Accumulators_,
    /// The iterator to load C from global memory.
    typename GlobalLoadIteratorC_,
    /// The transformer for C.
    typename GlobalTransformerC_,
    /// The transformer for D.
    typename GlobalTransformerD_,
    /// The iterator to store D to global memory.
    typename GlobalStoreIteratorD_,
    /// The iterator to store D to shared memory.
    typename SharedStoreIteratorD_,
    /// The shared store transformer for D.
    typename SharedStoreTransformerD_,
    /// The stream to load D from shared memory.
    typename SharedLoadStreamD_,
    /// The number of iterations in the epilogue.
    typename Iterations_,
    /// The iterations strides.
    typename Delta_,
    /// The functor to be used in the epilogue.
    typename Functor_,
    /// The index.
    typename Index_ = int>
struct GemmEpilogueTraits {
  //
  /// The output tile.
  typedef OutputTile_ OutputTile;
  /// The number of iterations.
  /// The accumulators.
  typedef Accumulators_ Accumulators;
  /// The iterator for C in global memory.
  typedef GlobalLoadIteratorC_ GlobalLoadIteratorC;
  /// The transformer for C.
  typedef GlobalTransformerC_ GlobalTransformerC;
  /// The transformer for D.
  typedef GlobalTransformerD_ GlobalTransformerD;
  /// The iterator for D in global memory.
  typedef GlobalStoreIteratorD_ GlobalStoreIteratorD;
  /// The iterator to store D in shared memory.
  typedef SharedStoreIteratorD_ SharedStoreIteratorD;
  /// The shared store transformer for D.
  typedef SharedStoreTransformerD_ SharedStoreTransformerD;
  /// The stream to store D in shared memory.
  typedef SharedLoadStreamD_ SharedLoadStreamD;
  /// typedef typename GemmConfig::EpilogueIterations Iterations;
  typedef Iterations_ Iterations;
  /// The iterations strides.
  typedef Delta_ Delta;

  /// The functor in charge of the math.
  typedef Functor_ Functor;
  /// The index.
  typedef Index_ Index;
  /// The long index
  typedef long long LongIndex;

  /// We do not support 3D or 4D shapes.
  static_assert(Iterations::kD == 1 && Iterations::kC == 1, "Unsupported 3D/4D shapes");

  /// The scalar.
  typedef typename Functor::Scalar Scalar;
  /// The scalar for C.
  typedef typename GlobalLoadIteratorC::Scalar ScalarC;
  /// The scalar for D.
  typedef typename GlobalStoreIteratorD::Scalar ScalarD;

  /// The params.
  struct Params {
    /// The strides for H and W in the different iterations of the epilogue.
    Index stride_h, stride_w;
    /// The params for the C iterator.
    typename GlobalLoadIteratorC::Params iterator_c;

    /// Batch stride for C matrix
    LongIndex batch_stride_C;

    /// The params for the D global iterator.
    typename GlobalStoreIteratorD::Params iterator_d;

    /// Batch stride for C matrix
    LongIndex batch_stride_D;

    /// The params for the D shared store iterator.
    typename SharedStoreIteratorD::Params shared_store_iterator_d;
    /// The params for the D shared load stream.
    typename SharedLoadStreamD::Params shared_load_stream_d;
    /// The functor params.
    typename Functor::Params functor;

    /// Setup the params.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {

      // The parameters for the functor.
      int error_code = functor.initialize(desc);
      if (error_code) {
        return error_code;
      }

      // At the end of the H iteration, we jump over a number of columns.
      this->stride_h = desc.D.leading_dim() * Delta::kH;
      // Nothing to do here.
      this->stride_w = 0;
      // Setup the params for the global memory iterator for C.
      error_code = iterator_c.initialize(desc.C.data(),
                                         desc.C.leading_dim(),
                                         desc.C.leading_dim(),
                                         desc.problem_size[1],
                                         stride_w,
                                         Delta::kW);

      batch_stride_C = desc.batch_stride_C;

      if (error_code) {
        return error_code;
      }

      // Setup the params for the global memory iterator for D.
      error_code = iterator_d.initialize(desc.D.data(),
                                   desc.D.leading_dim(),
                                   desc.D.leading_dim(),
                                   desc.problem_size[1],
                                   stride_w,
                                   Delta::kW);

      batch_stride_D = desc.batch_stride_D;

      return error_code;
    }
  };

  /// The shared memory storage to exchange data.
  union StreamSharedStorage {
    // The storage for the store iterator.
    typename SharedStoreIteratorD::SharedStorage store;
    // The storage for the store iterator.
    typename SharedLoadStreamD::SharedStorage load;
  };

  /// The shared memory to swizzle the data in the epilogue.
  struct SharedStorage {
    // The storage for the shared stream D.
    StreamSharedStorage shared_stream;

    //
    //
    //

    CUTLASS_DEVICE
    ScalarD* data() { return reinterpret_cast<ScalarD*>(&shared_stream.load); }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_, typename EpilogueFunctor_, typename Index_ = int>
struct GemmEpilogueTraitsHelper {
  /// The scalar.
  typedef typename EpilogueFunctor_::Scalar Scalar;
  /// The output tile.
  typedef typename GemmConfig_::OutputTile OutputTile;

  /// The number of iterations in the epilogue.
  typedef Shape<1,
                GemmConfig_::MultiplyAdd::AccumulatorsPerThread::kH /
                    GemmConfig_::kAccumulatorsPerLdsB,
                GemmConfig_::kAccumulatorsPerLdsB>
      Iterations;
  // The iteration strides in the H/W dimension.
  typedef Shape<0,
                GemmConfig_::kAccumulatorsPerLdsB*(
                    GemmConfig_::Warps::kH* GemmConfig_::MultiplyAdd::ThreadsPerWarp::kH - 1),
                0>
      Delta;
  /// The functor to do the math in the epilogue.
  typedef EpilogueFunctor_ Functor;

  /// The traits class to build the iterator to store to shared memory for D.
  typedef GemmSharedStoreTileDTraits<
      // The pointer is float.
      // typename Functor::Scalar,
      // Functor::Scalar is alpha, beta type, in mixed precision, alpha and beta may not be the same with accumulation.
      // In this case Functor::ScalarAccum is needed
      typename Functor::ScalarAccum,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The number of scalars per STS.
      GemmConfig_::kScalarsPerStsD,
      // The skew -- 128 / sizeof(ScalarD) / kScalarsPerStsD is the number of threads involved in
      // a single STS. We divide by 2 as our objective is to add a skew to the odd threads to
      // avoid bank conflicts between odd and even threads.
      128 / sizeof(typename GemmConfig_::ScalarD) / GemmConfig_::kScalarsPerStsD / 2 *
          GemmConfig_::kScalarsPerStsD>
      SharedStoreTileTraits;

  /// The iterator to store D to shared memory.
  typedef TileStoreIterator<SharedStoreTileTraits,
                            typename SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared>
      SharedStoreIteratorD;

  /// The shared store transformer for D.
  typedef Copy<typename SharedStoreIteratorD::Fragment> SharedStoreTransformerD;

  /// The traits class to build the iterator to load from shared memory for D.
  typedef GemmSharedLoadTileDTraits<
      // The pointer is float.
      // typename Functor::Scalar,
      // Functor::Scalar is alpha, beta type, in mixed precision, alpha and beta may not be the same with accumulation.
      // In this case Functor::ScalarAccum is needed
      typename Functor::ScalarAccum,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The number of columns of the output tile written by iteration.
      GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsD,
      // The skew.
      SharedStoreTileTraits::kSkew>
      SharedLoadTileTraits;

  /// The iterator to load D from shared memory.
  typedef TileLoadIterator<SharedLoadTileTraits,
                           typename SharedLoadTileTraits::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorD;
  /// The stream to load D.
  typedef SharedLoadStream<SharedLoadIteratorD> SharedLoadStreamD;

  /// The traits class to build the iterator to load data from global memory for C^N.
  typedef GemmGlobalTileCdTraits<
      // The pointer is float const.
      typename GemmConfig_::ScalarC const,
      // The tile has size (N / Iterations)xM in GEMM's terminology.
      Shape<1,
            GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // How many elements do we jump over at each iteration?
      Iterations::kW,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgC>
      GlobalLoadTileTraits;

  /// The iterator to load C.
  typedef GemmGlobalIteratorCd<GlobalLoadTileTraits, Index_> GlobalLoadIteratorC;
  /// The transformer for C.
  typedef Copy<typename GlobalLoadIteratorC::Fragment> GlobalTransformerC;

  /// The traits class to build the iterator to store data to global memory for D^N.
  typedef GemmGlobalTileCdTraits<
      // The pointer is float.
      typename GemmConfig_::ScalarD,
      // The tile has size (N / Iterations)xM in GEMM's terminology.
      Shape<1,
            GemmConfig_::OutputTile::kH / ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // How many elements do we jump over at each iteration?
      Iterations::kW,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerStgD>
      GlobalStoreTileTraits;

  /// The iterator to store D.
  typedef GemmGlobalIteratorCd<GlobalStoreTileTraits, Index_> GlobalStoreIteratorD;
  /// The transformer for D.
  typedef Copy<typename GlobalStoreIteratorD::Fragment> GlobalTransformerD;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The GEMM config.
    typename GemmConfig_,
    /// The epilogue functor to do the math in the epilogue.
    typename EpilogueFunctor_,
    /// The index.
    typename Index_ = int,
    /// The helper to create the traits class.
    typename Helper_ = GemmEpilogueTraitsHelper<GemmConfig_, EpilogueFunctor_, Index_> >
struct SimplifiedGemmEpilogueTraits : public GemmEpilogueTraits<
                                          // The output tile.
                                          typename GemmConfig_::OutputTile,
                                          // The accumulators.
                                          typename GemmConfig_::Accumulators,
                                          // The global iterator for C.
                                          typename Helper_::GlobalLoadIteratorC,
                                          // The transformer for C.
                                          typename Helper_::GlobalTransformerC,
                                          // The transformer for D.
                                          typename Helper_::GlobalTransformerD,
                                          // The global iterator for D.
                                          typename Helper_::GlobalStoreIteratorD,
                                          // The iterator to store D to shared memory.
                                          typename Helper_::SharedStoreIteratorD,
                                          // The shared store transformer for D.
                                          typename Helper_::SharedStoreTransformerD,
                                          // The stream to load D from shared memory.
                                          typename Helper_::SharedLoadStreamD,
                                          // The number of iterations.
                                          typename Helper_::Iterations,
                                          // The strides between iterations.
                                          typename Helper_::Delta,
                                          // The functor to be used in the epilogue.
                                          EpilogueFunctor_,
                                          // The index.
                                          Index_> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
