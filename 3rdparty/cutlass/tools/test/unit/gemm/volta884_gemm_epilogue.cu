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
// clang-format off

#include <cublas_v2.h>
#include <cstring>
#include "cutlass_unit_test.h"

#include "tools/util/half.h"
#include "tools/util/host_matrix.h"
#include "tools/util/tensor_view_io.h"

#include "cutlass/tile_traits_standard.h"
#include "cutlass/gemm/linear_scaling.h"

#include "cutlass/gemm/volta884_multiplicand.h"
#include "cutlass/gemm/volta884_multiply_add.h"
#include "cutlass/gemm/mma_global_stream.h"
#include "cutlass/gemm/volta884_gemm_epilogue_traits.h"
#include "cutlass/gemm/volta884_shared_tile.h"
#include "cutlass/gemm/mma_shared_stream.h"
#include "cutlass/gemm/mma_epilogue.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

#if CUTLASS_ENABLE_TENSOR_CORE_MMA

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel that verifies the Volta884 epilogue against the naive epilogue implementation
template <typename EpilogueTraits, typename AccumulatorType>
__global__ void test_volta884_epilogue(
  typename EpilogueTraits::Params params,
  AccumulatorType *ptr_Product,
  int ldm,
  cutlass::Coord<3> problem_size) {

  // Shared memoryallocation
  __shared__ typename EpilogueTraits::SharedStorage shared_storage;

  // Construct the epilogue
  cutlass::gemm::MMAEpilogue<EpilogueTraits> epilogue(params, shared_storage, problem_size);

  // Initialize accumulators
  typedef typename EpilogueTraits::Accumulators Accumulators;

  typedef typename cutlass::gemm::Volta884NaiveEpilogue<
    AccumulatorType,
    typename EpilogueTraits::WarpDelta,
    cutlass::Shape<2,2,2,2> > NaiveEpilogue;

  Accumulators accumulators;

  // Artificially load accumulators with some random matrix product
  NaiveEpilogue naive(ptr_Product, ldm);
  naive.load(accumulators);

  // Store the accumulators
  epilogue.epilogue(accumulators);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ScalarC,
  /// Specifies the delta between warp accesses along the outer dimension
  typename WarpDelta
>
struct Volta884EpilogueTestbed {

  //
  // Type definitions
  //

  /// Warp-level tile
  typedef cutlass::Shape<4, 64, 64> WarpGemmTile;

  /// Thread-block scoped tile
  typedef typename cutlass::ShapeMul<
    WarpGemmTile,
    WarpDelta
  >::Shape OutputTile;

  /// Multiply-add operation
  typedef cutlass::gemm::Volta884MultiplyAdd<
    WarpGemmTile,
    cutlass::MatrixLayout::kColumnMajor,
    half,
    cutlass::MatrixLayout::kRowMajor,
    half,
    ScalarC
  > MultiplyAdd;

  //
  // Parameters for the epilogue
  //

  /// Epilogue functor
  typedef cutlass::gemm::LinearScaling<ScalarC> Functor;

  /// Traits for global tile access
  typedef cutlass::gemm::Volta884EpilogueGlobalTileTraits<
    WarpGemmTile,
    WarpDelta,
    1,
    ScalarC
  > EpilogueGlobalTileTraits;


  /// Defines traits for an epilogue of a Volta884 GEMM
  typedef cutlass::gemm::Volta884EpilogueTraits<
    OutputTile,
    WarpGemmTile,
    WarpDelta,
    typename MultiplyAdd::Accumulators,
    cutlass::gemm::Volta884SelectAccumulators<
      WarpGemmTile,
      WarpDelta,
      ScalarC
    >,
    cutlass::PredicatedTileLoadStream<
      cutlass::TileLoadIterator<
        EpilogueGlobalTileTraits,
        ScalarC,
        cutlass::IteratorAdvance::kH,
        cutlass::MemorySpace::kGlobal
      >,
      cutlass::gemm::Volta884EpiloguePredicateFunctor<EpilogueGlobalTileTraits, ScalarC>
    >,
    cutlass::PredicatedTileStoreStream<
      cutlass::TileStoreIterator<
        EpilogueGlobalTileTraits,
        ScalarC,
        cutlass::IteratorAdvance::kH,
        cutlass::MemorySpace::kGlobal
      >,
      cutlass::gemm::Volta884EpiloguePredicateFunctor<EpilogueGlobalTileTraits, ScalarC>
    >,
    cutlass::TileStoreStream<
      cutlass::gemm::Volta884EpilogueSharedStoreIterator<
        WarpGemmTile,
        WarpDelta,
        ScalarC,
        ScalarC
      >
    >,
    cutlass::TileLoadStream<
      cutlass::gemm::Volta884EpilogueSharedLoadIterator<
        WarpGemmTile,
        WarpDelta,
        ScalarC,
        1,
        ScalarC
      >
    >,
    Functor
  > EpilogueTraits;

  //
  //
  //

  /// Generates random elements
  template <typename T>
  struct RandomGenerator {
    RandomGenerator(
      int seed = -1
    ) { srand(seed); }

    T operator()() {
      int val = (rand() % 29) - 13;
      return T(val);
    }
  };

  typedef typename cutlass::TypeTraits<ScalarC>::host_type ScalarCHost;

  //
  // Data members
  //

  /// Input accumulator matrix
  cutlass::HostMatrix<ScalarCHost> tensor_C;

  /// Matrix product
  cutlass::HostMatrix<ScalarCHost> tensor_Product;

  /// Reference output
  cutlass::HostMatrix<ScalarCHost> tensor_Ref;

  /// Computed output
  cutlass::HostMatrix<ScalarCHost> tensor_D;

  //
  // Methods
  //

  Volta884EpilogueTestbed() {
    tensor_C.resize(OutputTile::kW, OutputTile::kH, cutlass::MatrixLayout::kColumnMajor);
    tensor_Product.resize(OutputTile::kW, OutputTile::kH, cutlass::MatrixLayout::kColumnMajor);
    tensor_Ref.resize(OutputTile::kW, OutputTile::kH, cutlass::MatrixLayout::kColumnMajor);
    tensor_D.resize_matrix(OutputTile::kW, OutputTile::kH, cutlass::MatrixLayout::kColumnMajor);
  }

  /// Runs a test case
  bool run() {

    tensor_C.fill_sequential();
    tensor_Product.fill_random(RandomGenerator<ScalarCHost>(17));

    tensor_D.fill(ScalarCHost(0));
    tensor_Ref.fill(ScalarCHost(0));

    tensor_C.sync_device();
    tensor_Product.sync_device();
    tensor_D.sync_device();

    // run kernel
    dim3 grid(1, 1);
    dim3 block(32 * cutlass::ShapeCount<WarpDelta>::kCount, 1, 1);

    typename EpilogueTraits::Params params;

    params.load_stream_c.iterator.initialize(
      tensor_C.device_data(),
      tensor_C.leading_dim(),
      tensor_C.leading_dim(),
      1);

    params.store_stream_d.iterator.initialize(
      tensor_D.device_data(),
      tensor_D.leading_dim(),
      tensor_D.leading_dim(),
      1);

    ScalarCHost alpha = 2;
    ScalarCHost beta = 1;

    params.functor.initialize(alpha, beta);

    cutlass::Coord<3> problem_size = cutlass::make_Coord(
        128,
        64 * EpilogueTraits::WarpDelta::kH - 7,
        64 * EpilogueTraits::WarpDelta::kW - 5);

    test_volta884_epilogue<EpilogueTraits, ScalarC><<< grid, block >>>(
      params,
      tensor_Product.device_data(),
      tensor_Product.leading_dim(),
      problem_size
    );

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy to host
    tensor_D.sync_host();

    // Compute reference based on alpha, beta, and the problem dimensions
    for (int j = 0; j < OutputTile::kH; ++j) {
      for (int i = 0; i < OutputTile::kW; ++i) {
        if (j < problem_size[1] && i < problem_size[2]) {
          tensor_Ref.host_data()[i + j * tensor_Ref.leading_dim()] =
            alpha * tensor_Product.host_data()[i + j * tensor_Product.leading_dim()] +
            beta * tensor_C.host_data()[i + j * tensor_C.leading_dim()];
        }
      }
    }

    // Verify result
    bool passed = tensor_D.bit_equals(tensor_Ref);

    if (!passed) {
      std::cout << "Mismatch:\n"
        << "Product = \n" << tensor_Product << "\n\n"
        << "C =\n" << tensor_C << "\n\n"
        << "Reference =\n" << tensor_Ref << "\n\n"
        << "D =\n" << tensor_D << std::endl;
    }

    return passed;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f32, 64x64x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 1, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}


TEST(volta884_epilogue_f32, 64x128x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 2, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}


TEST(volta884_epilogue_f32, 128x64x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 1, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

TEST(volta884_epilogue_f32, 128x128x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}


TEST(volta884_epilogue_f32, 256x128x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 2, 4, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

TEST(volta884_epilogue_f32, 128x256x32) {

  Volta884EpilogueTestbed<
    float,
    cutlass::Shape<1, 4, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 64x64x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 1, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 128x64x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 1, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 64x128x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 2, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 128x128x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 256x128x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 2, 4, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_epilogue_f16, 128x256x32) {

  Volta884EpilogueTestbed<
    half,
    cutlass::Shape<1, 4, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // CUTLASS_ENABLE_TENSOR_CORE_MMA

// clang-format on
