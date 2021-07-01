/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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

////////////////////////////////////////////////////////////////////////////////////////////////////

// Guard conditions around the entire file.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

////////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass_unit_tests.h"
#include "tools/util/half.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "cutlass/gemm/wmma_gemm_epilogue.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits, typename EpilogueTraits, typename LoadAccumulatorIterator>
__global__ void test_epilogue_kernel(
  typename EpilogueTraits::Params params,
  cutlass::Coord<3> problem,
  typename EpilogueTraits::AccumulatorScalar *accum_ptr,
  int ldm) {

  // Shared memory allocation
  __shared__ typename EpilogueTraits::SharedStorage shared_storage;

  //
  // Load accumulators from memory - normally, a GEMM would compute these
  //

  // Traits class defines tiling
  GemmTraits traits;

  int warp_id = (threadIdx.x / 32);
  cutlass::Coord<3> warp_offset = traits(warp_id);

  // Accumulator fragment
  typename EpilogueTraits::AccumulatorFragment accumulator;

  // Construct an out-of-band LoadIterator for accumulators to initialize them

  LoadAccumulatorIterator load_accum_iterator(accum_ptr, ldm, warp_offset);
  load_accum_iterator.load(accumulator);

  __syncthreads();

  //
  // Test the epilogue itself
  //

  typedef cutlass::gemm::WmmaGemmEpilogue<EpilogueTraits> Epilogue;

  Epilogue epilogue(params, problem, warp_offset);

  // Perform the epilogue operation
  epilogue.update(shared_storage, accumulator);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThreadBlockTile,
  typename WarpTile,
  typename WmmaTile,
  typename EpilogueTile,
  typename StreamTile,
  typename AccumulatorType,
  typename ScalarC
>
struct TestWmmaGemmEpilogue {

  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    AccumulatorType,
    AccumulatorType,
    1,
    AccumulatorType,
    EpilogueTile,
    StreamTile
  > Traits;

  // Construct an actual epilogue
  typedef cutlass::gemm::EpilogueLinearScaling<ScalarC, ScalarC, ScalarC, ScalarC> EpilogueLinearScaling;

  /// Define some traits
  typedef cutlass::gemm::WmmaGemmEpilogueTraitsBasic<
    ScalarC,
    typename Traits::WarpMultiplyAdd::StoreIteratorC,
    ScalarC,
    ThreadBlockTile,
    32 * Traits::Warps::kCount,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    EpilogueLinearScaling
  > WmmaGemmEpilogueTraits;

  /// Type alias for EpilogueTraits type
  typedef typename WmmaGemmEpilogueTraits::Traits EpilogueTraits;

  TestWmmaGemmEpilogue() {

  }

  void run(cutlass::Coord<3> problem) {
    //
    // Prepare accumulator tile
    //
    cutlass::HostTensor<ScalarC> accumulator_matrix;
    cutlass::HostTensor<ScalarC> source_matrix;
    cutlass::HostTensor<ScalarC> destination_matrix;

    accumulator_matrix.resize_matrix(
      ThreadBlockTile::kW,
      ThreadBlockTile::kH,
      cutlass::MatrixLayout::kColumnMajor);

    source_matrix.resize_matrix(
      problem[2],
      problem[1],
      cutlass::MatrixLayout::kColumnMajor);

    destination_matrix.resize_matrix(
      problem[2],
      problem[1],
      cutlass::MatrixLayout::kColumnMajor);

    accumulator_matrix.fill_sequential();

    source_matrix.fill_sequential();

    int value = 0;
    for (int row = 0; row < ThreadBlockTile::kW; ++row) {
      for (int col = 0; col < ThreadBlockTile::kH; ++col, ++value) {
        if (row < problem[2] && col < problem[1]) {
          source_matrix.at(cutlass::make_Coord(0, row, col, 0)) = ScalarC(value);
        }
      }
    }

    destination_matrix.fill(0);

    //
    // Launch test kernel
    //
    dim3 grid(1,1);
    dim3 block(32 * Traits::Warps::kCount, 1, 1);

    EpilogueLinearScaling functor;
    functor.initialize(1, 0);

    typename EpilogueTraits::Params params;

    params.initialize(
      functor,
      source_matrix.device_data(),
      source_matrix.leading_dim(),
      destination_matrix.device_data(),
      destination_matrix.leading_dim()
    );

    test_epilogue_kernel<
      Traits,
      EpilogueTraits,
      typename Traits::WarpMultiplyAdd::LoadIteratorC
    ><<< grid, block >>>(
      params,
      problem,
      accumulator_matrix.device_data(),
      accumulator_matrix.leading_dim()
    );

    destination_matrix.sync_host();

    EXPECT_TRUE(accumulator_matrix.bit_equals(destination_matrix))
      << "Accumulators:\n" << accumulator_matrix << "\nDestination:\n" << destination_matrix;
  }

  void run() {
    run(cutlass::make_Coord(ThreadBlockTile::kD, ThreadBlockTile::kH, ThreadBlockTile::kW));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Small epilogue
TEST(WmmaGemm_16x16x16, wmma_epilogue_basic) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 16, 16> ThreadBlockTile;
  typedef cutlass::Shape<16, 16, 16> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 16, 16> EpilogueTile;
  typedef cutlass::Shape<1, 16, 16> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run();
}

TEST(WmmaGemm_16x16x16, wmma_epilogue_ragged) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 16, 16> ThreadBlockTile;
  typedef cutlass::Shape<16, 16, 16> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 16, 16> EpilogueTile;
  typedef cutlass::Shape<1, 16, 16> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run(cutlass::make_Coord(0, 15, 15));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Small epilogue
TEST(WmmaGemm_32x32x16, wmma_epilogue_basic_32x32_32x32) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 32, 32> ThreadBlockTile;
  typedef cutlass::Shape<16, 32, 32> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 32, 32> EpilogueTile;
  typedef cutlass::Shape<1, 4, 32> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run();
}

/// Small epilogue
TEST(WmmaGemm_32x32x16, wmma_epilogue_basic_32x32_32x32_ragged) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 32, 32> ThreadBlockTile;
  typedef cutlass::Shape<16, 32, 32> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 32, 32> EpilogueTile;
  typedef cutlass::Shape<1, 4, 32> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run(cutlass::make_Coord(0, 14, 17));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Small epilogue
TEST(WmmaGemm_32x32x16, wmma_epilogue_basic_32x32_16x16) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 32, 32> ThreadBlockTile;
  typedef cutlass::Shape<16, 16, 16> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 32, 32> EpilogueTile;
  typedef cutlass::Shape<1, 4, 32> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run();
}

/// Small epilogue
TEST(WmmaGemm_32x32x16, wmma_epilogue_basic_32x32_16x16_ragged) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 32, 32> ThreadBlockTile;
  typedef cutlass::Shape<16, 16, 16> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 32, 32> EpilogueTile;
  typedef cutlass::Shape<1, 4, 32> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run(cutlass::make_Coord(0, 23, 19));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Large epilogue
TEST(WmmaGemm_128x128x16, wmma_epilogue_basic_32x32_16x16) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 128, 128> ThreadBlockTile;
  typedef cutlass::Shape<16, 32, 64> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 64, 64> EpilogueTile;
  typedef cutlass::Shape<1, 4, 64> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  typedef cutlass::gemm::WmmaGemmEpilogueStructure<
    ThreadBlockTile,
    EpilogueTile,
    StreamTile,
    WarpTile,
    WmmaTile
  > Structure;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run();
}

/// Large epilogue
TEST(WmmaGemm_128x128x16, wmma_epilogue_basic_32x32_16x16_ragged) {

  // GEMM threadblock structure
  typedef cutlass::Shape<16, 128, 128> ThreadBlockTile;
  typedef cutlass::Shape<16, 32, 64> WarpTile;
  typedef cutlass::Shape<16, 16, 16> WmmaTile;

  // Epilogue shapes
  typedef cutlass::Shape<1, 64, 64> EpilogueTile;
  typedef cutlass::Shape<1, 4, 64> StreamTile;

  typedef float AccumulatorType;
  typedef float ScalarC;

  typedef cutlass::gemm::WmmaGemmEpilogueStructure<
    ThreadBlockTile,
    EpilogueTile,
    StreamTile,
    WarpTile,
    WmmaTile
  > Structure;

  TestWmmaGemmEpilogue<
    ThreadBlockTile,
    WarpTile,
    WmmaTile,
    EpilogueTile,
    StreamTile,
    AccumulatorType,
    ScalarC
  >().run(cutlass::make_Coord(0, 119, 101));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // end guard conditional on SM70
