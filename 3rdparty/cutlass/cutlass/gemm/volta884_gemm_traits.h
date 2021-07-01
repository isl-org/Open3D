/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *   * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without specific prior written
 *     permission.
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
  \brief Defines structural properties for GEMM targeting Volta's mma.sync instruction
*/

#pragma once

// clang-format off

#include "cutlass/gemm/clear_accumulators.h"
#include "cutlass/gemm/gemm_config.h"
#include "cutlass/gemm/gemm_global_stream.h"
#include "cutlass/gemm/gemm_stream_pair.h"
#include "cutlass/gemm/threadblock_swizzle.h"
#include "cutlass/gemm/linear_scaling.h"
#include "cutlass/kernel_launch.h"

#include "cutlass/gemm/gemm_desc.h"
#include "cutlass/gemm/volta884_multiplicand.h"
#include "cutlass/gemm/volta884_multiply_add.h"
#include "cutlass/gemm/mma_global_stream.h"
#include "cutlass/gemm/mma_shared_stream.h"
#include "cutlass/gemm/volta884_gemm_epilogue_traits.h"
#include "cutlass/gemm/mma_epilogue.h"
#include "cutlass/gemm/gemm_mainloop.h"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines configuration for Volta884 GEMM
template <
    /// The layout for A.
    MatrixLayout::Kind LayoutA,
    /// The layout for B.
    MatrixLayout::Kind LayoutB,
    /// The tile size for the GEMM KxNxM.
    typename OutputTile_,
    /// Tile size for warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_,
    /// The accumulator type.
    typename Accumulator_,
    /// The source matrix type type.
    typename ScalarC_,
    /// The destination matrix type
    typename ScalarD_,
    /// Number of stages in shared memory
    int StageCount,

    /// If true, kernel is launched with CUDA launch bounds specified
    bool kLaunchBounds = true,
    /// If true, residue is computed in mainloop. If false, separate loops are instantiated.
    bool kResidueSeparate = true,
    /// Is residue performed in prologue?
    bool kResidueInProlog = false>
struct Volta884GemmConfig : public GemmConfig<
                                /// The scalar type for A.
                                half,
                                /// The scalar type for B.
                                half,
                                /// The scalar type for C.
                                ScalarC_,
                                /// The scalar type for D.
                                ScalarD_,
                                /// The threadblock tile size
                                OutputTile_,
                                /// The functor to do the math in the main loop.
                                Volta884MultiplyAdd<WarpGemmShape_,
                                                    LayoutA,
                                                    half,
                                                    LayoutB,
                                                    half,
                                                    Accumulator_>,
                                /// The number of scalars per LDG for A.
                                8,
                                /// The number of scalars per STS for A.
                                8,
                                /// The number of scalars per LDS for A.
                                8,
                                /// The number of scalars per LDG for B.
                                8,
                                /// The number of scalars per STS for B.
                                8,
                                /// The number of scalars per LDS for B.
                                8,
                                /// The number of scalars per LDG for C and STG for D.
                                16 / int(sizeof(ScalarD_)),
                                /// The number of scalars per STS for D.
                                16 / int(sizeof(ScalarD_)),
                                /// The number of scalars per LDS for D.
                                16 / int(sizeof(ScalarD_)),
                                /// The number of stages in shared memory.
                                StageCount,
                                /// If true, separate mainloop is instantiated
                                kResidueSeparate,
                                /// If true, compute residue in prolog
                                kResidueInProlog,
                                /// Launch bounds not used
                                kLaunchBounds> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines components of Volta884 GEMM
template <
  /// The layout for A.
  MatrixLayout::Kind LayoutA,
  /// The layout for B.
  MatrixLayout::Kind LayoutB,
  /// The tile size for the GEMM KxNxM.
  typename OutputTile_,
  /// Tile size for warp-level GEMM (K-by-N-by-M)
  typename WarpGemmShape_,
  /// The accumulator type.
  typename Accumulator_,
  /// The input matrix type type.
  typename ScalarC_,
  /// The output matrix type type.
  typename ScalarD_,
  /// Number of buffers in shared memory to use
  int StageCount,
  /// The functor to do the math in the epilogue.
  typename EpilogueFunctor_ = LinearScaling<Accumulator_>,
  /// The block swizzle to reorganize the grid.
  typename BlockSwizzle_ = IdentityBlockSwizzle,
  /// Selectively enables launch bounds
  bool LaunchBounds = false
>
struct Volta884GemmTraits {
  /// This traits
  typedef Volta884GemmTraits<
    LayoutA,
    LayoutB,
    OutputTile_,
    WarpGemmShape_,
    Accumulator_,
    ScalarC_,
    ScalarD_,
    StageCount,
    EpilogueFunctor_,
    BlockSwizzle_,
    LaunchBounds> This_;
  /// The struct that consumes this Traits
  typedef typename cutlass::gemm::GemmMainloop<This_> KernelClass;

  /// Layout of multiplicand A matrix
  static MatrixLayout::Kind const kLayoutA = LayoutA;

  /// Layout of multiplicand B matrix
  static MatrixLayout::Kind const kLayoutB = LayoutB;

  /// Dimensions of threadblock tile (concept Shape)
  typedef OutputTile_ OutputTile;

  /// Shape of warp-level accumulators
  typedef WarpGemmShape_ WarpGemmShape;

  /// Multiplicand A scalar type
  typedef half ScalarA;

  /// Multiplicand B scalar type
  typedef half ScalarB;

  /// Data type of internal accumulator
  typedef Accumulator_ Accumulator;

  /// Data type of input accumulator matrix operand
  typedef ScalarC_ ScalarC;

  /// Data type of output accumulator matrix operand
  typedef ScalarD_ ScalarD;

  /// Shape of individual mma.sync instruction
  typedef Shape<4, 16, 16> InstructionShape;

  /// Tile size for an individual warp-level multiply-add
  typedef Shape<InstructionShape::kD, WarpGemmShape::kH, WarpGemmShape::kW> WarpTile;

  /// Defines properties about GEMM needed by host code
  typedef Volta884GemmConfig<kLayoutA,
                             kLayoutB,
                             OutputTile,
                             WarpGemmShape,
                             Accumulator,
                             ScalarC,
                             ScalarD,
                             StageCount,
                             LaunchBounds>
      GemmConfig;

  //
  // Derived types
  //

  /// Index type
  typedef int Index;

  /// Partitioning of threadblock into warps
  typedef typename ShapeDiv<OutputTile, WarpGemmShape>::Shape WarpDelta;

  /// Number of warps per threadblock
  static int const kWarpCount = ShapeCount<WarpDelta>::kCount;

  /// Defines iterators for A matrix
  typedef Volta884Multiplicand<GemmOperand::kA, kLayoutA, OutputTile, WarpTile, kWarpCount, WarpDelta>
      MultiplicandA;

  /// Defines iterators for B matrix
  typedef Volta884Multiplicand<GemmOperand::kB, kLayoutB, OutputTile, WarpTile, kWarpCount, WarpDelta>
      MultiplicandB;

  //
  // GemmTraits mandatory type definitions
  //

  /// Maps hardware threadblocks to logical partitions of the GEMM
  typedef BlockSwizzle_ BlockSwizzle;

  /// Clears accumulators
  typedef ClearAccumulators<ScalarC> ClearAccumulators;

  /// Loads multiplicands from global memory
  typedef GlobalLoadStreamPair<
      MMAGlobalLoadStream<GemmOperand::kA,
                           kLayoutA,
                           typename MultiplicandA::LoadIterator,
                           Copy<typename MultiplicandA::LoadIterator::Fragment>,
                           typename MultiplicandA::StoreIterator,
                           StageCount>,
      MMAGlobalLoadStream<GemmOperand::kB,
                           kLayoutB,
                           typename MultiplicandB::LoadIterator,
                           Copy<typename MultiplicandB::LoadIterator::Fragment>,
                           typename MultiplicandB::StoreIterator,
                           StageCount>,
                           GemmConfig::kResidueInProlog >
      GlobalLoadStream;

  /// Memory needed to store the threadblock-scoped GEMM tile
  typedef typename GlobalLoadStream::ThreadblockTileStorage ThreadblockTileStorage;
  union MainLoopStorage {

    /// Stores the threadblock tile
    ThreadblockTileStorage threadblock_tile;

    /// Storage for GEMM global stream
    typename GlobalLoadStream::SharedStorage global_to_shared_stream;
  };

  /// Loads multiplicands from shared memory
  typedef SharedStreamPair<
      MMASharedLoadStream<typename MultiplicandA::WarpLoadIterator,
                           Copy<typename MultiplicandA::WarpLoadIterator::Fragment>,
                           StageCount>,
      MMASharedLoadStream<typename MultiplicandB::WarpLoadIterator,
                           Copy<typename MultiplicandB::WarpLoadIterator::Fragment>,
                           StageCount> >
      SharedStream;

  // Multiply-add object specialized for Volta mma.sync
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;

#if 0
  /// Naive epilogue for updating the output matrix
  typedef cutlass::gemm::Volta884NaiveEpilogue<ScalarC,
                                               typename MultiplicandA::WarpDelta,
                                               typename MultiplyAdd::Iterations>
      Epilogue;
#else

  /// Efficient epilogue
  typedef cutlass::gemm::MMAEpilogue<
    typename Volta884GemmEpilogueTraitsHelper<
      GemmConfig,
      EpilogueFunctor_
    >::EpilogueTraits
  > Epilogue;

#endif

  /// Parameters structure
  struct Params : public KernelLaunchConfiguration {
    /// The dimensions of the GEMM.
    GemmCoord problem_size;

    /// The K range for every partition except the last one
    int partitionK_range;

    /// The params for the global load stream
    typename GlobalLoadStream::Params global_to_shared_stream;

    /// The params for the shared load stream
    typename SharedStream::Params shared_stream;

    /// The params for the epilogue.
    typename Epilogue::Params epilogue;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() {}

    /// Initialize the parameters.
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE Params(GemmDesc_ const& desc) {
      initialize(desc);
    }

    /// Initialize the Params struct
    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {

      // Problem size
      problem_size = desc.problem_size;

      // there is no partitionK in the default case
      partitionK_range = problem_size[0];
      // Compute grid dimensions
      BlockSwizzle block_swizzle;
      this->block = dim3(GemmConfig::kThreads);
      this->grid = block_swizzle.get_grid_layout(
        problem_size,
        make_Coord_from_shape<OutputTile>());

      // Compute offset to residue
      Index gemm_k = problem_size[0];
      Index offset_to_residue = (gemm_k % OutputTile::kD) ? gemm_k - (gemm_k % OutputTile::kD) : 0;
      Index offset_to_residue_last_partition = (partitionK_range % OutputTile::kD) ? partitionK_range - (partitionK_range % OutputTile::kD) : 0;
      // Initialize parameters objects for
      global_to_shared_stream.stream_a.initialize(
        desc.A,
        desc.batch_stride_A,
        offset_to_residue,
        offset_to_residue_last_partition);

      global_to_shared_stream.stream_b.initialize(
        desc.B,
        desc.batch_stride_B,
        offset_to_residue,
        offset_to_residue_last_partition);

      // The epilogue.
      epilogue.initialize(desc);
      return 0;
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
        make_Coord(k, n, m, batch_count),
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
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& partitonK_desc, Index partitionK_count_, Index partitionK_multiple_ = 1) {
      // partitionK GEMM is a specialized batched stried gemm with different K ranges per batch
      // the problem_size of each batch is (lastK_size, n, m)
      // add more comments here
      // the k range for every batch excpet the last one

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
        desc.A,
        desc.batch_stride_A,
        offset_to_residue,
        offset_to_residue_last_partition
      );
      if (error_code) {
        return error_code;
      }

      error_code = global_to_shared_stream.stream_b.initialize(
        desc.B,
        desc.batch_stride_B,
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

  /// Shared memory storage
  union SharedStorage {
    /// Storage required during mainloop phase
    MainLoopStorage main_loop;

    /// Shared storage needed for epilogue
    typename Epilogue::SharedStorage epilogue;
  };

  /// The memory fence for shared loads.
  static CUTLASS_DEVICE void shared_load_fence(bool in_loop) {
    if (StageCount < 2) {
        __syncthreads();
    }
  }

  /// The memory fence for shared stores.
  static CUTLASS_DEVICE void shared_store_fence(bool in_loop) {
      __syncthreads();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

// clang-format on
