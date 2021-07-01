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

#include <cublas_v2.h>
#include <cstring>
#include "cutlass_unit_test.h"

#include "tools/util/half.h"
#include "tools/util/host_matrix.h"
#include "tools/util/tensor_view_io.h"

#include "cutlass/gemm/volta884_multiplicand.h"
#include "cutlass/gemm/volta884_multiply_add.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

#if CUTLASS_ENABLE_TENSOR_CORE_MMA

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simplified GEMM: computes one threadblock-scoped matrix product.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to verify a tile of data loaded from GMEM, stored to SMEM, and loaded into RF computes
/// the expected mma.sync product
template <
  typename MultiplicandA,
  typename MultiplicandB,
  typename ScalarC
>
__global__ void test_volta884_matrix_product(
  typename MultiplicandA::LoadIterator::Params load_A_params,
  typename MultiplicandB::LoadIterator::Params load_B_params,
  float *C,
  int ldc,
  int active_k_idx) {

  // Define thread-block scoped load iterators
  typename MultiplicandA::LoadIterator load_A_iterator(load_A_params);
  typename MultiplicandB::LoadIterator load_B_iterator(load_B_params);


  // Define shared memory buffers
  static int const kSmemAElements =
    cutlass::ShapeCount<typename MultiplicandA::StoreIterator::OperandShape>::kCount;

  static int const kSmemBElements =
    cutlass::ShapeCount<typename MultiplicandB::StoreIterator::OperandShape>::kCount;

  __shared__ uint16_t smem_A_buffer[kSmemAElements];
  __shared__ uint16_t smem_B_buffer[kSmemBElements];


  // Instantiate thread-block-scoped store iterators
  typename MultiplicandA::StoreIterator::Params store_A_params(reinterpret_cast<half *>(&smem_A_buffer[0]));
  typename MultiplicandB::StoreIterator::Params store_B_params(reinterpret_cast<half *>(&smem_B_buffer[0]));

  typename MultiplicandA::StoreIterator store_A_iterator(store_A_params);
  typename MultiplicandB::StoreIterator store_B_iterator(store_B_params);


  // Load thread-block scoped fragments
  typename MultiplicandA::LoadIterator::Fragment threadblock_A_frag;
  typename MultiplicandB::LoadIterator::Fragment threadblock_B_frag;

  __syncthreads();

  // A operand
  load_A_iterator.load(threadblock_A_frag);
  store_A_iterator.store(threadblock_A_frag);

  // Barrier to  enforce SMEM consistency
  __syncthreads();

  // B operand
  load_B_iterator.load(threadblock_B_frag);
  store_B_iterator.store(threadblock_B_frag);


  // Barrier to  enforce SMEM consistency
  __syncthreads();

  // Instantiate warp-scoped load iterators
  typename MultiplicandA::WarpLoadIterator::Params warp_A_params(reinterpret_cast<half const *>(&smem_A_buffer[0]));
  typename MultiplicandB::WarpLoadIterator::Params warp_B_params(reinterpret_cast<half const *>(&smem_B_buffer[0]));

  typename MultiplicandA::WarpLoadIterator warp_load_A(warp_A_params);
  typename MultiplicandB::WarpLoadIterator warp_load_B(warp_B_params);

  // Instantiate a multiply-add object specialized for Volta mma.sync
  typedef cutlass::gemm::Volta884MultiplyAdd<
    typename MultiplicandA::WarpTile,
    MultiplicandA::kLayout,
    half,
    MultiplicandB::kLayout,
    half,
    ScalarC
  > MultiplyAdd;

  typedef cutlass::gemm::Volta884NaiveEpilogue<
    ScalarC,
    typename MultiplicandA::WarpDelta,
    typename MultiplyAdd::Iterations
  > NaiveEpilogue;

  MultiplyAdd multiply_add;
  NaiveEpilogue epilogue(C, ldc);

  // Initialize accumulator fragment
  typename MultiplyAdd::Accumulators accumulators;


  for (int i = 0; i < MultiplyAdd::Accumulators::kElements; ++i) {
    accumulators[i] = threadIdx.x;
  }

  epilogue.clear(accumulators);

  // Iterate over the K dimension of the threadblock tile
  #pragma unroll
  for (int k_idx = 0; k_idx < MultiplicandA::Tile::kD / MultiplyAdd::WarpTile::kD; ++k_idx) {

    if (active_k_idx < 0 || active_k_idx == k_idx) {
      typename MultiplicandA::WarpLoadIterator::Fragment warp_A_frag;
      typename MultiplicandB::WarpLoadIterator::Fragment warp_B_frag;

      // Load warp-scoped fragments
      warp_load_A.load(warp_A_frag, cutlass::make_Coord(k_idx, 0, 0, 0));
      warp_load_B.load(warp_B_frag, cutlass::make_Coord(k_idx, 0, 0, 0));

      // Compute accumulated matrix product
      multiply_add.multiply_add(warp_A_frag, warp_B_frag, accumulators, accumulators);
    }
  }

  // Store accumulator tile
  epilogue.store(accumulators);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Identifies multiplicand of GEMM (A or B)
  cutlass::MatrixLayout::Kind LayoutA,
  /// Specifies layout of data in source memory
  cutlass::MatrixLayout::Kind LayoutB,
  /// Accumulator type
  typename ScalarC,
  /// Specifies threadblock tile shape
  typename Tile,
  /// Specifies the warp tile shape
  typename WarpTile,
  /// Specifies the number of participating warps
  int WarpCount,
  /// Specifies the delta between warp accesses along the outer dimension
  typename WarpDelta
>
struct Volta884MatrixProductTestbed {

  //
  // Type definitions
  //

  typedef cutlass::gemm::Volta884Multiplicand<
    cutlass::GemmOperand::kA,
    LayoutA,
    Tile,
    WarpTile,
    WarpCount,
    WarpDelta> MultiplicandA;

  typedef cutlass::gemm::Volta884Multiplicand<
    cutlass::GemmOperand::kB,
    LayoutB,
    Tile,
    WarpTile,
    WarpCount,
    WarpDelta> MultiplicandB;

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

  /// Depth of an mma.sync instruction
  static int const kWarpK = 4;

  //
  // Data members
  //

  cutlass::HostMatrix<cutlass::half_t> tensor_A;
  cutlass::HostMatrix<cutlass::half_t> tensor_B;
  cutlass::HostMatrix<ScalarC> tensor_C;
  cutlass::HostMatrix<ScalarC> tensor_Ref;

  //
  // Methods
  //

  Volta884MatrixProductTestbed() {

    tensor_A.resize(cutlass::make_Coord(Tile::kW, Tile::kD), LayoutA);
    tensor_B.resize(cutlass::make_Coord(Tile::kD, Tile::kH), LayoutB);
    tensor_C.resize(cutlass::make_Coord(Tile::kW, Tile::kH), cutlass::MatrixLayout::kColumnMajor);
    tensor_Ref.resize(cutlass::make_Coord(Tile::kW, Tile::kH), cutlass::MatrixLayout::kColumnMajor);

  }

  /// Runs a test case
  bool run_once(int seed, int active_k_idx = -1) {

  #if 0
    // For debugging, it helps to see sequential elements
    tensor_A.fill_sequential();
    tensor_B.fill_identity();
  #else
    // Fill with random elements
    tensor_A.fill_random(RandomGenerator<cutlass::half_t>(seed + 53));
    tensor_B.fill_random(RandomGenerator<cutlass::half_t>(seed + 97));
  #endif

    if (active_k_idx >= 0) {
      // overwrite all but the active k index with zeros
      int const m_stride = (LayoutA == cutlass::MatrixLayout::kRowMajor ? Tile::kD : 1);
      int const a_k_stride = (LayoutA == cutlass::MatrixLayout::kRowMajor ? 1 : Tile::kW);

      int const n_stride = (LayoutB == cutlass::MatrixLayout::kRowMajor ? 1 : Tile::kD);
      int const b_k_stride = (LayoutB == cutlass::MatrixLayout::kRowMajor ? Tile::kH : 1);

      for (int k_idx = 0; k_idx < Tile::kD / kWarpK; ++k_idx) {
        if (active_k_idx != k_idx) {

          for (int k = 0; k < kWarpK; ++k) {
            for (int m = 0; m < Tile::kW; ++m) {
              tensor_A.host_data()[m_stride * m + a_k_stride * (k_idx * kWarpK + k)] = 0;
            }
            for (int n = 0; n < Tile::kH; ++n) {
              tensor_B.host_data()[n_stride * n + b_k_stride * (k_idx * kWarpK + k)] = 0;
            }
          }
        }
      }
    }

    tensor_A.sync_device();
    tensor_B.sync_device();

    tensor_C.fill(ScalarC(0));
    tensor_Ref.fill(ScalarC(0));
    tensor_C.sync_device();

    // run kernel
    dim3 grid(1, 1);
    dim3 block(32 * WarpCount, 1, 1);

    typename MultiplicandA::LoadIterator::Params load_A_params(
      tensor_A.device_data(),
      tensor_A.leading_dim() * 8,
      tensor_A.leading_dim(),
      8
    );

    typename MultiplicandB::LoadIterator::Params load_B_params(
      tensor_B.device_data(),
      tensor_B.leading_dim() * 8,
      tensor_B.leading_dim(),
      8
    );

    test_volta884_matrix_product<MultiplicandA, MultiplicandB, ScalarC><<< grid, block >>>(
      load_A_params,
      load_B_params,
      tensor_C.device_data(),
      tensor_C.leading_dim(),
      active_k_idx
    );

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy to host
    tensor_C.sync_host();

    // Compute reference
    cutlass::reference::host::Gemm(
      cutlass::gemm::GemmCoord(
        tensor_A.size().column(),
        tensor_Ref.size().column(),
        tensor_Ref.size().row()),
      ScalarC(1),
      tensor_A,
      tensor_B,
      ScalarC(0),
      tensor_Ref,
      ScalarC(0));

    // Assert bit-level equivalence
    bool passed = tensor_Ref.bit_equals(tensor_C);

    EXPECT_TRUE(passed)
      << "Incorrect matrix product\n"
      << "A =\n" << tensor_A
      << "\nB =\n" << tensor_B
      << "\nRef =\n" << tensor_Ref
      << "\nMMA=\n" << tensor_C;

    return passed;
  }

  /// Executes a set of test cases containing unique, randomly chosen matrices and verifies
  /// bit equivalence with the reference implementation.
  bool run(int test_count = 16) {

    bool passed = true;

  #if 1
    // Run several tests with deterministic seeds
    for (int i = 0; i < test_count && passed; ++i) {
      passed = run_once(i * 41 + i * 17);
    }

  #else
    // For debugging, run the full matrix product with exactly one K-index non-zero
    for (int k_idx = 0; passed && k_idx < Tile::kD / kWarpK; ++k_idx) {
      passed = run_once(17, k_idx);
      if (!passed) {
        std::cout << "Failed on k_idx = " << k_idx
          << "  [" << k_idx * kWarpK << ".." << (k_idx + 1) * kWarpK - 1 << "]" << std::endl;
      }
    }
  #endif

    return passed;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 64x64x32, 128x64x32, 64x128x32, 128x128x32, 256x128x32, 128x256x32, 64x64x128
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Congruous loading
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 64x64x32_32x32x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<4, 32, 32>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 128x64x32_64x32x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<4, 32, 64>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 64x128x32_32x64x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<4, 64, 32>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 64x64x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<4, 64, 64>,
    1,
    cutlass::Shape<1, 1, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 64x64x128) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<128, 64, 64>,
    cutlass::Shape<4, 64, 64>,
    1,
    cutlass::Shape<1, 1, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 128x64x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<4, 64, 64>,
    2,
    cutlass::Shape<1, 1, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 64x128x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<4, 64, 64>,
    2,
    cutlass::Shape<1, 2, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 128x128x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<4, 64, 64>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 256x128x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 128, 256>,
    cutlass::Shape<4, 64, 64>,
    8,
    cutlass::Shape<1, 2, 4, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_nt, 128x256x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    float,
    cutlass::Shape<32, 256, 128>,
    cutlass::Shape<4, 64, 64>,
    8,
    cutlass::Shape<1, 4, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Crosswise loading
//
////////////////////////////////////////////////////////////////////////////////////////////////////


TEST(volta884_matrix_product_tn, 64x64x32_32x32x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<4, 32, 32>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 128x64x32_64x32x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<4, 32, 64>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 64x128x32_32x64x4) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<4, 64, 32>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 64x64x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<4, 64, 64>,
    1,
    cutlass::Shape<1, 1, 1, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 128x64x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<4, 64, 64>,
    2,
    cutlass::Shape<1, 1, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 128x128x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<4, 64, 64>,
    4,
    cutlass::Shape<1, 2, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 256x128x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 128, 256>,
    cutlass::Shape<4, 64, 64>,
    8,
    cutlass::Shape<1, 2, 4, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(volta884_matrix_product_tn, 128x256x32) {

  Volta884MatrixProductTestbed<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    float,
    cutlass::Shape<32, 256, 128>,
    cutlass::Shape<4, 64, 64>,
    8,
    cutlass::Shape<1, 4, 2, 1>
  > testbed;

  EXPECT_TRUE(testbed.run());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // if defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
