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
    \brief Defines structural properties of complete GEMM computation.
*/
#pragma once

#include "cutlass/convert.h"
#include "cutlass/matrix_traits.h"
#include "cutlass/reshape_tile.h"
#include "cutlass/tile_allocation.h"
#include "cutlass/tile_iterator.h"
#include "cutlass/kernel_launch.h"

#include "cutlass/gemm/clear_accumulators.h"
#include "cutlass/gemm/gemm_config.h"
#include "cutlass/gemm/gemm_desc.h"
#include "cutlass/gemm/gemm_stream_pair.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_operand.h"
#include "cutlass/gemm/gemm_shared_stream.h"
#include "cutlass/gemm/threadblock_swizzle.h"
#include "cutlass/gemm/gemm_mainloop.h"
namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind, typename GemmConfig_>
struct GemmTileTraitsHelperA {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct GemmTileTraitsHelperA<MatrixLayout::kColumnMajor, GemmConfig_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// The traits class to build the iterator to load data from global memory for A^N.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is column-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxM in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA>
      GlobalTileTraits;

  /// The traits class to build the iterator to store data to shared memory for A^N.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer is float.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kW * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsA>
      SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for A^N.
  typedef GemmSharedLoadTileATraits<
      // The pointer is float const.
      MultiplyAddScalar const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsA,
      // The skew.
      0>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct GemmTileTraitsHelperA<MatrixLayout::kRowMajor, GemmConfig_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarA Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarA MultiplyAddScalar;

  /// The traits class to build the iterator to load data from global memory for A^T.
  typedef GemmGlobalTileTraits<
      // That's A.
      GemmOperand::kA,
      // A is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size MxK in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kW, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgA>
      GlobalTileTraits;

  /// The number of scalars in 4B.
  static int const kScalarsIn4B = sizeof(MultiplyAddScalar) > 4 ? 1 : 4 / sizeof(MultiplyAddScalar);
  /// The skew for A.
  static int const kSkewA = 128 / sizeof(MultiplyAddScalar) / GemmConfig_::kScalarsPerStsA /
                            GlobalTileTraits::Threads::kW * kScalarsIn4B;

  /// The traits class to build the iterator to store data to shared memory for A^T.
  typedef GemmSharedStoreWithSkewTileAbTraits <
      // The pointer is float.
      MultiplyAddScalar,
      // The tile has size KxM in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kW * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS.
      GemmConfig_::kScalarsPerStsA,
      // The skew to avoid bank conflicts added in the tile W dimension.
      kSkewA<GemmConfig_::kScalarsPerLdsA ? GemmConfig_::kScalarsPerLdsA : kSkewA>
          SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for A^T.
  typedef GemmSharedLoadTileATraits<
      // The pointer is float const.
      MultiplyAddScalar const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsA,
      // The skew.
      SharedStoreTileTraits::kSkew>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <enum MatrixLayout::Kind, typename GemmConfig_>
struct GemmTileTraitsHelperB {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct GemmTileTraitsHelperB<MatrixLayout::kColumnMajor, GemmConfig_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kColumnMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// The traits class to build the iterator to load data from global memory for B^N.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // B is column-major.
      MatrixLayout::kColumnMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size MxK in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kH, GemmConfig_::OutputTile::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      Shape<1, GemmConfig_::kThreads / GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kD>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB>
      GlobalTileTraits;

  /// The number of scalars in 4B.
  static int const kScalarsIn4B = sizeof(MultiplyAddScalar) > 4 ? 1 : 4 / sizeof(MultiplyAddScalar);
  /// The skew for B.
  static int const kSkewB = 128 / sizeof(MultiplyAddScalar) / GemmConfig_::kScalarsPerStsB /
                            GlobalTileTraits::Threads::kW * kScalarsIn4B;

  /// The traits class to build the iterator to store data to shared memory for B^N.
  typedef GemmSharedStoreWithSkewTileAbTraits <
      // The pointer is float.
      MultiplyAddScalar,
      // The tile has size KxN in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kH * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as (threads / K) x K (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS.
      GemmConfig_::kScalarsPerStsB,
      // The skew to avoid bank conflicts added in the tile W dimension.
      kSkewB<GemmConfig_::kScalarsPerLdsB ? GemmConfig_::kScalarsPerLdsB : kSkewB>
          SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for B^N.
  typedef GemmSharedLoadTileBTraits<
      // The pointer is float const.
      MultiplyAddScalar const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsB,
      // The skew.
      SharedStoreTileTraits::kSkew>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig_>
struct GemmTileTraitsHelperB<MatrixLayout::kRowMajor, GemmConfig_> {
  /// The layout.
  static MatrixLayout::Kind const kLayout = MatrixLayout::kRowMajor;

  /// The input scalar.
  typedef typename GemmConfig_::ScalarB Scalar;
  /// The scalar stored in shared memory.
  typedef typename GemmConfig_::MultiplyAdd::ScalarB MultiplyAddScalar;

  /// The traits class to build the iterator to load data from global memory for B^T.
  typedef GemmGlobalTileTraits<
      // That's B.
      GemmOperand::kB,
      // B is row-major.
      MatrixLayout::kRowMajor,
      // The pointer is float const.
      Scalar const,
      // The tile has size KxN in GEMM's terminology.
      Shape<1, GemmConfig_::OutputTile::kD, GemmConfig_::OutputTile::kH>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      Shape<1, ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerLdgB>
      GlobalTileTraits;

  /// The traits class to build the iterator to store data to shared memory for B^T.
  typedef GemmSharedStoreTileAbTraits<
      // The pointer is float.
      MultiplyAddScalar,
      // The tile has size KxN in GEMM's terminology.
      Shape<GemmConfig_::kStages,
            GemmConfig_::OutputTile::kD / GemmConfig_::InstructionShape::kD,
            GemmConfig_::OutputTile::kH * GemmConfig_::InstructionShape::kD>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      typename GlobalTileTraits::Threads,
      // The number of scalars per STS (STS.32 or STS.128, etc).
      GemmConfig_::kScalarsPerStsB>
      SharedStoreTileTraits;

  /// The traits class to build the iterator to load from shared memory for B^T.
  typedef GemmSharedLoadTileBTraits<
      // The pointer is float const.
      MultiplyAddScalar const,
      // The output tile size.
      typename GemmConfig_::OutputTile,
      // The number of warps.
      typename GemmConfig_::Warps,
      // The number of threads per warp.
      typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
      // The shape of the FMA instruction.
      typename GemmConfig_::InstructionShape,
      // The number of stages.
      GemmConfig_::kStages,
      // The number of scalars per LDS.
      GemmConfig_::kScalarsPerLdsB,
      // The skew.
      0>
      SharedLoadTileTraits;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The GEMM configuration.
    typename GemmConfig_,
    /// The stream to load A from global memory to shared memory.
    typename GlobalLoadStreamA_,
    /// The stream to load B from global memory to shared memory.
    typename GlobalLoadStreamB_,
    /// The stream to load A from shared memory.
    typename SharedLoadStreamA_,
    /// The stream to load B from shared memory.
    typename SharedLoadStreamB_,
    /// The epilogue.
    typename Epilogue_,
    /// The block swizzle to reorganize the grid.
    typename BlockSwizzle_ = IdentityBlockSwizzle,
    /// The index.
    typename Index_ = int,
    /// The tool used to clear accumulators.
    typename ClearAccumulators_ = ClearAccumulators<typename GemmConfig_::Accumulators::Element> >

struct GemmTraits {
  /// This traits
  typedef GemmTraits<GemmConfig_,
    GlobalLoadStreamA_,
    GlobalLoadStreamB_,
    SharedLoadStreamA_,
    SharedLoadStreamB_,
    Epilogue_,
    BlockSwizzle_,
    Index_,
    ClearAccumulators_> This_;

  /// The struct that consumes this Traits
  typedef typename cutlass::gemm::GemmMainloop<This_> KernelClass;

  /// The configuration.
  typedef GemmConfig_ GemmConfig;
  /// The output tile.
  typedef typename GemmConfig::OutputTile OutputTile;

  /// The stream to load A from global memory to shared memory.
  typedef GlobalLoadStreamA_ GlobalLoadStreamA;
  /// The layout of A.
  static MatrixLayout::Kind const kLayoutA = GlobalLoadStreamA::kLayout;
  /// The scalar for A.
  typedef typename GlobalLoadStreamA_::Scalar ScalarA;

  /// The stream to load B from global memory to shared memory.
  typedef GlobalLoadStreamB_ GlobalLoadStreamB;
  /// The layout of B.
  static MatrixLayout::Kind const kLayoutB = GlobalLoadStreamB::kLayout;
  /// The scalar for B.
  typedef typename GlobalLoadStreamB_::Scalar ScalarB;

  /// The iterator for A to load from shared memory.
  typedef SharedLoadStreamA_ SharedLoadStreamA;
  /// The iterator for B to load from shared memory.
  typedef SharedLoadStreamB_ SharedLoadStreamB;

  /// The multiply-add functor.
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;
  /// The epilogue.
  typedef Epilogue_ Epilogue;
  /// The scalars in the epilogue.
  typedef typename Epilogue::ScalarC ScalarC;
  typedef typename Epilogue::ScalarD ScalarD;

  /// The block swizzle to reorganize the grid.
  typedef BlockSwizzle_ BlockSwizzle;
  /// The index.
  typedef Index_ Index;
  /// Clear the accumulators.
  typedef ClearAccumulators_ ClearAccumulators;

  /// Assemble the global load streams for A/B.
  typedef GlobalLoadStreamPair<GlobalLoadStreamA,
                               GlobalLoadStreamB,
                               GemmConfig::kResidueInProlog>
      GlobalLoadStream;

  /// Memory needed to store the threadblock-scoped GEMM tile
  typedef typename GlobalLoadStream::ThreadblockTileStorage ThreadblockTileStorage;

  /// Assemble the shared load streams for A/B.
  typedef SharedStreamPair<SharedLoadStreamA, SharedLoadStreamB> SharedStream;

  /// Parameters object constructable on the host.
  struct Params : public KernelLaunchConfiguration {

    /// GEMM problem size
    GemmCoord problem_size;

    /// The K range for every partition except the last one
    int partitionK_range;

    /// Parameters object for the global load stream
    typename GlobalLoadStream::Params global_to_shared_stream;

    /// Parameters object for the shared load stream
    typename SharedStream::Params shared_stream;

    /// The params for the epilogue.
    typename Epilogue::Params epilogue;

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      // Set the problem size.
      problem_size = desc.problem_size;

      // there is no partitionK in the default case
      partitionK_range = problem_size[0];
      // Compute grid dimensions
      BlockSwizzle block_swizzle;
      this->block = dim3(GemmConfig::kThreads);
      this->grid = block_swizzle.get_grid_layout(
        problem_size,
        make_Coord_from_shape<OutputTile>());

      // Compute offset to residue.
      // partitionK_range <= problem_size[0]
      Index gemm_k = problem_size[0];
      Index offset_to_residue_last_partition = (gemm_k % OutputTile::kD) ? gemm_k - (gemm_k % OutputTile::kD) : 0;
      Index offset_to_residue = (partitionK_range % OutputTile::kD) ? partitionK_range - (partitionK_range % OutputTile::kD) : 0;

      // Initialize parameters objects for
      int error_code = global_to_shared_stream.stream_a.initialize(
        desc.A.data(),
        desc.batch_stride_A,
        desc.A.leading_dim(),
        offset_to_residue,
        offset_to_residue_last_partition
      );
      if (error_code) {
        return error_code;
      }

      error_code = global_to_shared_stream.stream_b.initialize(
        desc.B.data(),
        desc.batch_stride_B,
        desc.B.leading_dim(),
        offset_to_residue,
        offset_to_residue_last_partition
      );

      if (error_code) {
        return error_code;
      }

      // The epilogue.
      return epilogue.initialize(desc);
    }

    /// Helper to construct a GEMM params using a BLAS-like API
    CUTLASS_HOST_DEVICE int initialize(Index m,
                                       Index n,
                                       Index k,
                                       typename Epilogue::Scalar alpha,
                                       ScalarA const* d_a,
                                       Index lda,
                                       ScalarB const* d_b,
                                       Index ldb,
                                       typename Epilogue::Scalar beta,
                                       ScalarC const* d_c,
                                       Index ldc,
                                       ScalarD* d_d,
                                       Index ldd) {
      GemmDesc<ScalarA, ScalarB, ScalarC, ScalarD, typename Epilogue::Scalar> desc(
        GemmCoord(k, n, m, 1),
        alpha,
        TensorRef<ScalarA const, 2>(d_a, lda),
        TensorRef<ScalarB const, 2>(d_b, ldb),
        beta,
        TensorRef<ScalarC const, 2>(d_c, ldc),
        TensorRef<ScalarD, 2>(d_d, ldd)
      );

      return this->initialize(desc);
    }

    /// Helper to construct a batched GEMM params
    CUTLASS_HOST_DEVICE int initialize(Index m,
                                       Index n,
                                       Index k,
                                       typename Epilogue::Scalar alpha,
                                       ScalarA const* d_a,
                                       Index lda,
                                       long long int batch_stride_A,
                                       ScalarB const* d_b,
                                       Index ldb,
                                       long long int batch_stride_B,
                                       typename Epilogue::Scalar beta,
                                       ScalarC const* d_c,
                                       Index ldc,
                                       long long int batch_stride_C,
                                       ScalarD* d_d,
                                       Index ldd,
                                       long long int batch_stride_D,
                                       Index batch_count) {
      GemmDesc<ScalarA, ScalarB, ScalarC, ScalarD, typename Epilogue::Scalar> desc(
        GemmCoord(k, n, m, batch_count),
        alpha,
        TensorRef<ScalarA const, 2>(d_a, lda),
        batch_stride_A,
        TensorRef<ScalarB const, 2>(d_b, ldb),
        batch_stride_B,
        beta,
        TensorRef<ScalarC const, 2>(d_c, ldc),
        batch_stride_C,
        TensorRef<ScalarD, 2>(d_d, ldd),
        batch_stride_D
      );

      return this->initialize(desc);
    }

    /// Helper to construct a partitionedK GEMM params
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& partitonK_desc,
      Index partitionK_count_,
      Index partitionK_multiple_ = 1 // each partition will be mulitples of partitionK_multiple_
    ) {
      // partitionK GEMM is a specialized batched stried gemm with different K ranges per batch
      // the problem_size of each batch is (lastK_size, n, m)
      // add more comments here
      // the k range for every batch excpet the last one
      //assert(partitionK_count_ > 0);
      partitionK_range = partitonK_desc.problem_size.k() / partitionK_count_;
      partitionK_range = partitionK_range - (partitionK_range % partitionK_multiple_);
      // the k range of the last batch
      // int lastK_range = (partitonK_desc.problem_size.k() % partitionK_range) + partitionK_range;
      int lastK_range = partitonK_desc.problem_size.k() - partitionK_range * (partitionK_count_ - 1);

      assert((partitionK_range % partitionK_multiple_) == 0);
      assert(partitionK_range > 0);
      assert((lastK_range % partitionK_multiple_) == 0);
      assert(lastK_range > 0);

      int k_size = lastK_range;
      int lda = partitonK_desc.A.stride(0);
      int ldb = partitonK_desc.B.stride(0);
      int ldc = partitonK_desc.C.stride(0);
      int ldd = partitonK_desc.D.stride(0);
      int n = partitonK_desc.problem_size.n();


      long long int batch_stride_A = (kLayoutA == cutlass::MatrixLayout::kColumnMajor) ? lda * partitionK_range : partitionK_range;
      long long int batch_stride_B = (kLayoutB == cutlass::MatrixLayout::kColumnMajor) ? partitionK_range : partitionK_range * ldb;
      long long int batch_stride_C = ldc * n;
      long long int batch_stride_D = ldd * n;

      GemmDesc<ScalarA, ScalarB, ScalarC, ScalarD, typename Epilogue::Scalar> desc(
        //we pass lastK_size as per batch K. there is also a range that will match partitionK_size
        GemmCoord(k_size, partitonK_desc.problem_size.n(), partitonK_desc.problem_size.m(), partitionK_count_),
        partitonK_desc.alpha,
        partitonK_desc.A,
        batch_stride_A,
        partitonK_desc.B,
        batch_stride_B,
        partitonK_desc.beta,
        partitonK_desc.C,
        batch_stride_C,
        partitonK_desc.D,
        batch_stride_D
      );

      // Set the problem size.
      problem_size = desc.problem_size;

      // Compute grid dimensions
      BlockSwizzle block_swizzle;
      this->block = dim3(GemmConfig::kThreads);
      this->grid = block_swizzle.get_grid_layout(
        problem_size,
        make_Coord_from_shape<OutputTile>());

      // Compute offset to residue.
      // partitionK_range <= problem_size[0]
      Index gemm_k = problem_size[0];
      Index offset_to_residue_last_partition = (gemm_k % OutputTile::kD) ? gemm_k - (gemm_k % OutputTile::kD) : 0;
      Index offset_to_residue = (partitionK_range % OutputTile::kD) ? partitionK_range - (partitionK_range % OutputTile::kD) : 0;

      // Initialize parameters objects for
      int error_code = global_to_shared_stream.stream_a.initialize(
        desc.A.data(),
        desc.batch_stride_A,
        desc.A.leading_dim(),
        offset_to_residue,
        offset_to_residue_last_partition
      );
      if (error_code) {
        return error_code;
      }

      error_code = global_to_shared_stream.stream_b.initialize(
        desc.B.data(),
        desc.batch_stride_B,
        desc.B.leading_dim(),
        offset_to_residue,
        offset_to_residue_last_partition
      );

      if (error_code) {
        return error_code;
      }

      // The epilogue.
      return epilogue.initialize(desc);
    }


    /// Helper to construct a partitionedK GEMM params
    CUTLASS_HOST_DEVICE int initialize(Index m,
                                       Index n,
                                       Index k,
                                       typename Epilogue::Scalar alpha,
                                       ScalarA const* d_a,
                                       Index lda,
                                       ScalarB const* d_b,
                                       Index ldb,
                                       typename Epilogue::Scalar beta,
                                       ScalarC const* d_c,
                                       Index ldc,
                                       ScalarD* d_d,
                                       Index ldd,
                                       Index partitionK_count_,
                                       Index partitionK_multiple_ = 1) {

      GemmDesc<ScalarA, ScalarB, ScalarC, ScalarD, typename Epilogue::Scalar> desc(
        GemmCoord(k, n, m, 1),
        alpha,
        TensorRef<ScalarA const, 2>(d_a, lda),
        TensorRef<ScalarB const, 2>(d_b, ldb),
        beta,
        TensorRef<ScalarC const, 2>(d_c, ldc),
        TensorRef<ScalarD, 2>(d_d, ldd)
      );


      return this->initialize(desc, partitionK_count_, partitionK_multiple_);
    }
  };

  // The storage for the main loop + prologue.
  struct MainLoopSharedStorage {
    /// Stores the threadblock tile
    ThreadblockTileStorage threadblock_tile;

    /// Storage for GEMM global stream
    typename GlobalLoadStream::SharedStorage global_to_shared_stream;

    /// Storage for clearing accumulators
    typename ClearAccumulators::SharedStorage clear;
  };

  /// The storage in shared memory.
  union SharedStorage {
    // The storage for the main loop.
    MainLoopSharedStorage main_loop;
    // The storage for the epilogue.
    typename Epilogue::SharedStorage epilogue;
  };

  /// The memory fence for shared loads.
  static CUTLASS_DEVICE void shared_load_fence(bool in_loop) {
    if (SharedLoadStreamA::Iterator::kRequiresLoadFence ||
        SharedLoadStreamB::Iterator::kRequiresLoadFence) {
        __syncthreads();
    }
  }

  /// The memory fence for shared stores.
  static CUTLASS_DEVICE void shared_store_fence(bool in_loop) {
      __syncthreads();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTileTraitsHelperA_, typename GemmTileTraitsHelperB_, typename Index_>
struct SimplifiedGemmTraitsHelper {
  /// The global iterator to load A from global memory.
  typedef GemmGlobalIteratorAb<typename GemmTileTraitsHelperA_::GlobalTileTraits, Index_>
      GlobalLoadIteratorA;
  /// The data converter for A before storing to shared memory.
  typedef Copy<typename GlobalLoadIteratorA::Fragment> GlobalTransformerA;
  /// The iterator to store A to shared memory.
  typedef TileStoreIterator<typename GemmTileTraitsHelperA_::SharedStoreTileTraits,
                            typename GemmTileTraitsHelperA_::SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared>
      SharedStoreIteratorA;
  /// The stream to load A from global memory to shared memory.
  typedef GlobalLoadStream<GemmOperand::kA,
                              GlobalLoadIteratorA,
                              SharedStoreIteratorA,
                              GlobalTransformerA>
      GlobalLoadStreamA;

  /// The global iterator to load B from global memory.
  typedef GemmGlobalIteratorAb<typename GemmTileTraitsHelperB_::GlobalTileTraits, Index_>
      GlobalLoadIteratorB;
  /// The data converter for B before storing to shared memory.
  typedef Copy<typename GlobalLoadIteratorB::Fragment> GlobalTransformerB;
  /// The iterator to store B to shared memory.
  typedef TileStoreIterator<typename GemmTileTraitsHelperB_::SharedStoreTileTraits,
                            typename GemmTileTraitsHelperB_::SharedStoreTileTraits::Scalar,
                            IteratorAdvance::kH,
                            MemorySpace::kShared>
      SharedStoreIteratorB;
  /// The stream to load B from global memory to shared memory.
  typedef GlobalLoadStream<GemmOperand::kB,
                              GlobalLoadIteratorB,
                              SharedStoreIteratorB,
                              GlobalTransformerB>
      GlobalLoadStreamB;

  /// The iterator to load A from shared memory.
  typedef TileLoadIterator<typename GemmTileTraitsHelperA_::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperA_::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorA;
  /// The stream to load A from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorA> SharedLoadStreamA;
  /// The iterator to load B from shared memory.
  typedef TileLoadIterator<typename GemmTileTraitsHelperB_::SharedLoadTileTraits,
                           typename GemmTileTraitsHelperB_::Scalar,
                           IteratorAdvance::kH,
                           MemorySpace::kShared>
      SharedLoadIteratorB;
  /// The stream to load B from shared memory.
  typedef SharedLoadStream<SharedLoadIteratorB> SharedLoadStreamB;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The layout for A.
    MatrixLayout::Kind kLayoutA_,
    /// The layout for B.
    MatrixLayout::Kind kLayoutB_,
    /// The config for the GEMM.
    typename GemmConfig_,
    /// The epilogue.
    typename Epilogue_,
    /// The index.
    typename Index_ = int,
    // The configuration for the A matrix.
    typename GemmTileTraitsHelperA_ = GemmTileTraitsHelperA<kLayoutA_, GemmConfig_>,
    // The configuration for the B matrix.
    typename GemmTileTraitsHelperB_ = GemmTileTraitsHelperB<kLayoutB_, GemmConfig_>,
    // The helper class to create the streams and iterators.
    typename Helper_ =
        SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA_, GemmTileTraitsHelperB_, Index_> >
struct SimplifiedGemmTraits : public GemmTraits<
                                  // The config.
                                  GemmConfig_,
                                  // The stream to load A from global memory to shared memory.
                                  typename Helper_::GlobalLoadStreamA,
                                  // The stream to load B from global memory to shared memory.
                                  typename Helper_::GlobalLoadStreamB,
                                  // The stream to load A from shared memory.
                                  typename Helper_::SharedLoadStreamA,
                                  // The stream to load B from shared memory.
                                  typename Helper_::SharedLoadStreamB,
                                  // The epilogue.
                                  Epilogue_,
                                  // The block swizzle to reorganize the grid.
                                  IdentityBlockSwizzle,
                                  // The index.
                                  Index_,
                                  // The tool used to clear accumulators.
                                  ClearAccumulators<typename GemmConfig_::Accumulators::Element> > {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
