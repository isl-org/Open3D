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
  \brief Defines structural properties for complex-valued GEMM targeting Volta's mma.sync
  instruction.

    At present, it expects split complex representation in global memory in which the real part and
    imaginary parts of a complex-valued matrices are disjoint (a structure of arrays). This is in
    contrast with an interleaved complex representation which is an array of structures.
*/

#pragma once

// clang-format off

#include "cutlass/gemm/clear_accumulators.h"
#include "cutlass/gemm/gemm_config.h"
#include "cutlass/gemm/gemm_stream_pair.h"
#include "cutlass/gemm/threadblock_swizzle.h"
#include "cutlass/gemm/linear_scaling.h"
#include "cutlass/kernel_launch.h"
#include "cutlass/tensor_ref_collection.h"

#include "cutlass/gemm/gemm_desc.h"

#include "cutlass/gemm/volta884_multiplicand.h"
#include "cutlass/gemm/mma_shared_stream.h"
#include "cutlass/gemm/volta884_gemm_traits.h"

#include "cutlass/gemm/volta884_complex_multiply_add.h"
#include "cutlass/gemm/volta884_complex_global_stream.h"
#include "cutlass/gemm/volta884_complex_shared_stream.h"
#include "cutlass/gemm/volta884_complex_gemm_epilogue_traits.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines configuration for Volta884 GEMM
template <
    /// The layout for A.
    MatrixLayout::Kind LayoutA,
    /// Indicates matrix transform on multiplicand A
    MatrixTransform::Kind TransformA,
    /// The layout for B.
    MatrixLayout::Kind LayoutB,
    /// Indicates matrix transform on multiplicand B
    MatrixTransform::Kind TransformB,
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
    /// Enables or disables launch bounds
    bool LaunchBounds>
struct Volta884ComplexGemmConfig : public GemmConfig<
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
                                Volta884ComplexMultiplyAdd<WarpGemmShape_,
                                                    LayoutA,
                                                    TransformA,
                                                    half,
                                                    LayoutB,
                                                    TransformB,
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
                                true,
                                /// If true, compute residue in prolog
                                false,
                                /// Launch bounds not used
                                LaunchBounds> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines components of Volta884 GEMM
template <
  /// The layout for A.
  MatrixLayout::Kind LayoutA,
  /// Indicates matrix transform on multiplicand A
  MatrixTransform::Kind TransformA,
  /// The layout for B.
  MatrixLayout::Kind LayoutB,
  /// Indicates matrix transform on multiplicand B
  MatrixTransform::Kind TransformB,
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
  typename EpilogueFunctor_ = SplitComplexLinearScaling<Accumulator_>,
  /// Enables or disables launch bounds
  bool LaunchBounds = false
>
struct Volta884ComplexGemmTraits {

  /// This is insane.
  typedef Volta884ComplexGemmTraits<
    LayoutA,
    TransformA,
    LayoutB,
    TransformB,
    OutputTile_,
    WarpGemmShape_,
    Accumulator_,
    ScalarC_,
    ScalarD_,
    StageCount,
    EpilogueFunctor_,
    LaunchBounds> This;

  /// The actual device-side GEMM
  typedef GemmMainloop<This> KernelClass;

  /// Layout of multiplicand A matrix
  static MatrixLayout::Kind const kLayoutA = LayoutA;

  /// If true, A operand is conjugated
  static MatrixTransform::Kind const kTransformA = TransformA;

  /// Layout of multiplicand B matrix
  static MatrixLayout::Kind const kLayoutB = LayoutB;

  /// If true, B operand is conjugated
  static MatrixTransform::Kind const kTransformB = TransformB;

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
  typedef Volta884ComplexGemmConfig<
                             kLayoutA,
                             kTransformA,
                             kLayoutB,
                             kTransformB,
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

  /// Long index type
  typedef long long LongIndex;

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
  typedef IdentityBlockSwizzle BlockSwizzle;

  /// Clears accumulators
  typedef ClearAccumulators<ScalarC> ClearAccumulators;

  /// Loads multiplicands from global memory
  typedef GlobalLoadStreamPair<
      Volta884ComplexGlobalLoadStream<GemmOperand::kA,
                               kLayoutA,
                               typename MultiplicandA::LoadIterator,
                               Copy<typename MultiplicandA::LoadIterator::Fragment>,
                               typename MultiplicandA::StoreIterator,
                               StageCount>,
      Volta884ComplexGlobalLoadStream<GemmOperand::kB,
                               kLayoutB,
                               typename MultiplicandB::LoadIterator,
                               Copy<typename MultiplicandB::LoadIterator::Fragment>,
                               typename MultiplicandB::StoreIterator,
                               StageCount>,
                               GemmConfig::kResidueInProlog >
      GlobalLoadStream;

  /// Memory needed to store the threadblock-scoped GEMM tile
  typedef typename GlobalLoadStream::ThreadblockTileStorage ThreadblockTileStorage;

  /// Shared memory storage for mainloop phase
  union MainLoopStorage {

    /// Stores the threadblock tile
    ThreadblockTileStorage threadblock_tile;

    /// Storage for GEMM global stream
    typename GlobalLoadStream::SharedStorage global_to_shared_stream;
  };

  /// Loads multiplicands from shared memory
  typedef SharedStreamPair<
      Volta884ComplexSharedLoadStream<typename MultiplicandA::WarpLoadIterator,
                               Copy<typename MultiplicandA::WarpLoadIterator::Fragment>,
                               StageCount>,
      Volta884ComplexSharedLoadStream<typename MultiplicandB::WarpLoadIterator,
                               Copy<typename MultiplicandB::WarpLoadIterator::Fragment>,
                               StageCount> >
      SharedStream;

  // Multiply-add object specialized for Volta mma.sync
  typedef typename GemmConfig::MultiplyAdd MultiplyAdd;

  #if 0
  /// Naive epilogue for updating the output matrix
  typedef Volta884ComplexNaiveEpilogue<ScalarC,
                                               typename MultiplicandA::WarpDelta,
                                               typename MultiplyAdd::Iterations>
      Epilogue;

  #else

  /// Efficient epilogue
  typedef MMAEpilogue<
    Volta884ComplexGemmEpilogueTraits<GemmConfig, EpilogueFunctor_>
  > Epilogue;

  #endif

  /// Tensor reference to A multiplicand
  typedef ZipTensorRef<
    TensorRef<ScalarA, 2>,
    TensorRef<ScalarA, 2>
  > TensorRefA;

  /// Tensor reference to B multiplicand
  typedef ZipTensorRef<
    TensorRef<ScalarB, 2>,
    TensorRef<ScalarB, 2>
  > TensorRefB;

  /// Tensor reference to C multiplicand
  typedef ZipTensorRef<
    TensorRef<ScalarC, 2>,
    TensorRef<ScalarC, 2>
  > TensorRefC;

  /// Tensor reference to D multiplicand
  typedef ZipTensorRef<
    TensorRef<ScalarD, 2>,
    TensorRef<ScalarD, 2>
  > TensorRefD;

  /// gemm::ProblemDesc<>
  typedef GemmDesc<
    TensorRefA,
    TensorRefB,
    TensorRefC,
    TensorRefD,
    float
  > GemmDesc;

  /// Parameters structure
  struct Params : public KernelLaunchConfiguration {
    /// The dimensions of the GEMM.
    GemmCoord problem_size;

    /// PartitionK_range
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

    /// Initialize the Params struct
    CUTLASS_HOST_DEVICE int initialize(
      Index m,
      Index n,
      Index k,
      platform::complex<typename Epilogue::Scalar> alpha,
      ScalarA const* real_A,
      Index real_lda,
      ScalarA const* imag_A,
      Index imag_lda,
      ScalarB const* real_B,
      Index real_ldb,
      ScalarB const* imag_B,
      Index imag_ldb,
      platform::complex<typename Epilogue::Scalar> beta,
      ScalarC const* real_C,
      Index real_ldc,
      ScalarC const* imag_C,
      Index imag_ldc,
      ScalarD* real_D,
      Index real_ldd,
      ScalarD* imag_D,
      Index imag_ldd) {

      problem_size = make_Coord(k, n, m, 1);

      partitionK_range = problem_size.k();

      // Compute grid dimensions
      BlockSwizzle block_swizzle;
      this->block = dim3(GemmConfig::kThreads);
      this->grid = block_swizzle.get_grid_layout(
        problem_size,
        make_Coord_from_shape<OutputTile>());

      // Initialize global load streams
      global_to_shared_stream.stream_a.initialize(
        make_ZipTensorRef(
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(real_A, real_lda), 0),
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(imag_A, imag_lda), 0)
        ),
        0
      );

      global_to_shared_stream.stream_b.initialize(
        make_ZipTensorRef(
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(real_B, real_ldb), 0),
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(imag_B, imag_ldb), 0)
        ),
        0
      );

      return epilogue.initialize(
        alpha,
        beta,
        real_C,
        real_ldc,
        imag_C,
        imag_ldc,
        real_D,
        real_ldd,
        imag_D,
        imag_ldd
      );
    }

    /// Initialize the Params struct
    CUTLASS_HOST_DEVICE int initialize(
      Index m,
      Index n,
      Index k,
      platform::complex<typename Epilogue::Scalar> alpha,
      ScalarA const* real_A,
      Index real_lda,
      LongIndex batch_stride_A_real,
      ScalarA const* imag_A,
      Index imag_lda,
      LongIndex batch_stride_A_imag,
      ScalarB const* real_B,
      Index real_ldb,
      LongIndex batch_stride_B_real,
      ScalarB const* imag_B,
      Index imag_ldb,
      LongIndex batch_stride_B_imag,
      platform::complex<typename Epilogue::Scalar> beta,
      ScalarC const* real_C,
      Index real_ldc,
      LongIndex batch_stride_C_real,
      ScalarC const* imag_C,
      Index imag_ldc,
      LongIndex batch_stride_C_imag,
      ScalarD* real_D,
      Index real_ldd,
      LongIndex batch_stride_D_real,
      ScalarD* imag_D,
      Index imag_ldd,
      LongIndex batch_stride_D_imag,
      int batch_count) {

      problem_size = make_Coord(k, n, m, batch_count);
      partitionK_range = problem_size.k();

      // Compute grid dimensions
      BlockSwizzle block_swizzle;
      this->block = dim3(GemmConfig::kThreads);
      this->grid = block_swizzle.get_grid_layout(
        problem_size,
        make_Coord_from_shape<OutputTile>());

      // Initialize global load streams
      global_to_shared_stream.stream_a.initialize(
        make_ZipTensorRef(
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(real_A, real_lda), batch_stride_A_real),
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(imag_A, imag_lda), batch_stride_A_imag)
        ),
        0
      );

      global_to_shared_stream.stream_b.initialize(
        make_ZipTensorRef(
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(real_B, real_ldb), batch_stride_B_real),
          TensorRefBatchStrided<half const, 2>(TensorRef<half const, 2>(imag_B, imag_ldb), batch_stride_B_imag)
        ),
        0
      );

      return epilogue.initialize(
        alpha,
        beta,
        real_C,
        real_ldc,
        batch_stride_C_real,
        imag_C,
        imag_ldc,
        batch_stride_C_imag,
        real_D,
        real_ldd,
        batch_stride_D_real,
        imag_D,
        imag_ldd,
        batch_stride_D_imag
      );
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
  static CUTLASS_DEVICE void shared_store_fence(bool in_loop) { __syncthreads(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass

// clang-format on
