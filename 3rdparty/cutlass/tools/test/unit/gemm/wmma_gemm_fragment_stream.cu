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
#include "tools/util/tensor_view_io.h"
#include "tools/util/host_tensor.h"

#include "tools/test/unit/gemm/gemm_testbed.h"

#include "cutlass/gemm/gemm_fragment_stream.h"
#include "cutlass/gemm/warp_multiply_add_nvcuda.h"


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

template <typename FragmentStream>
__global__ void fragment_stream(typename FragmentStream::Params params, half *output) {

  __shared__ typename FragmentStream::Storage storage;

  params.store_params.initialize(storage);
  FragmentStream stream(
    params,
    cutlass::make_Coord(16, 256, 256)
  );

  // load
  stream.load();

  // store
  stream.commit();

  __syncthreads();

  // one thread writes it all out
  if (threadIdx.x == 0) {

    half const *ptr = reinterpret_cast<half const *>(storage.data());

    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < FragmentStream::Storage::Shape::kCount; ++i) {
      output[i] = ptr[i];
    }
  }
}

}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct TestGemmDesc {
  int m, n, k;
  inline __host__ __device__ TestGemmDesc() : m(0), n(0), k(0) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThreadBlockTile,
  cutlass::MatrixLayout::Kind LayoutA,
  cutlass::MatrixLayout::Kind LayoutB,
  int Threads,
  int ScalarsPerAccess
>
struct TestGemmFragmentStream {

  /// TileStream for Operand A
  typedef cutlass::gemm::GemmFragmentStreamTraits<
    cutlass::gemm::GemmOperand::kA,
    uint16_t,
    LayoutA,
    ThreadBlockTile,
    Threads,
    ScalarsPerAccess
  > FragmentStreamTraitsA;

  /// Defines fragment stream for A operand
  typedef typename cutlass::gemm::GemmFragmentStream<FragmentStreamTraitsA> FragmentStreamA;

  /// TileStream for Operand B
  typedef typename cutlass::gemm::GemmFragmentStreamTraits<
    cutlass::gemm::GemmOperand::kB,
    uint16_t,
    LayoutB,
    ThreadBlockTile,
    Threads,
    ScalarsPerAccess
  > FragmentStreamTraitsB;

  /// Defines fragment stream for A operand
  typedef typename cutlass::gemm::GemmFragmentStream<FragmentStreamTraitsB> FragmentStreamB;

  //
  // Data members
  //

  cutlass::HostTensor<cutlass::half_t> tensor_A_in;
  cutlass::HostTensor<cutlass::half_t> tensor_A_out;

  cutlass::HostTensor<cutlass::half_t> tensor_B_in;
  cutlass::HostTensor<cutlass::half_t> tensor_B_out;

  //
  // Methods
  //

  /// Constructor
  TestGemmFragmentStream() {
    tensor_A_in.resize_matrix(ThreadBlockTile::kW, ThreadBlockTile::kD, LayoutA);
    tensor_A_out.resize_matrix(ThreadBlockTile::kW, ThreadBlockTile::kD, LayoutA);

    tensor_B_in.resize_matrix(ThreadBlockTile::kD, ThreadBlockTile::kH, LayoutB);
    tensor_B_out.resize_matrix(ThreadBlockTile::kD, ThreadBlockTile::kH, LayoutB);
  }

  /// Writes details about TileStream
  template <typename TileStream>
  std::ostream & write(std::ostream &out, typename TileStream::Params const &params) {

    out << "TileStream::LoadIterator\n"
      << "  Tile(" << TileStream::LoadIterator::Tile::kH << ", "
      << TileStream::LoadIterator::Tile::kW << ")\n"
      << "  Delta(" << TileStream::LoadIterator::Steps::kH << ", "
      << TileStream::LoadIterator::Steps::kW << ")\n"
      << "  Iterations(" << TileStream::LoadIterator::Iterations::kH << ", "
      << TileStream::LoadIterator::Iterations::kW << ")\n";

    out
      << "  stride_h: " << params.load_params.stride_h << "\n"
      << "  stride_w: " << params.load_params.stride_w << "\n"
      << "  inc_d: " << params.load_params.inc_d << "\n"
      << "  inc_h: " << params.load_params.inc_h << "\n"
      << "  inc_w: " << params.load_params.inc_w << std::endl;

    out << "output elements: " << TileStream::Storage::Shape::kCount << std::endl;

    return out;
  }

  /// Runs test
  void run() {

    tensor_A_in.fill_linear(
        LayoutA == cutlass::MatrixLayout::kColumnMajor ?
        cutlass::make_Coord(1, 1, ThreadBlockTile::kW, 1) :
        cutlass::make_Coord(1, ThreadBlockTile::kD, 1, 1));

    tensor_A_out.fill(0);

    tensor_A_in.sync_device();
    tensor_A_out.sync_device();

    tensor_B_in.fill_linear(
        LayoutB == cutlass::MatrixLayout::kColumnMajor ?
        cutlass::make_Coord(1, 1, ThreadBlockTile::kD, 1) :
        cutlass::make_Coord(1, ThreadBlockTile::kH, 1, 1));

    tensor_B_out.fill(0);

    tensor_B_in.sync_device();
    tensor_B_out.sync_device();


    typename FragmentStreamA::Params params_A;
    typename FragmentStreamB::Params params_B;

    TestGemmDesc desc;
    params_A.initialize(
      desc,
      reinterpret_cast<uint16_t const *>(tensor_A_in.device_ref().data()),
      tensor_A_in.leading_dim()
    );

    params_B.initialize(
      desc,
      reinterpret_cast<uint16_t const *>(tensor_A_in.device_ref().data()),
      tensor_B_in.leading_dim()
    );

    test::fragment_stream<FragmentStreamA><<< dim3(1,1,1), dim3(Threads,1,1) >>>(
      params_A,
      tensor_A_out.device_data()
    );

    test::fragment_stream<FragmentStreamB><<< dim3(1,1,1), dim3(Threads,1,1) >>>(
      params_B,
      tensor_B_out.device_data()
    );

    tensor_A_out.sync_host();
    tensor_B_out.sync_host();

    bool passed_A = tensor_A_in.bit_equals(tensor_A_out);
    bool passed_B = tensor_B_in.bit_equals(tensor_B_out);

    EXPECT_TRUE(passed_A) << tensor_A_out;
    if (!passed_A) {
      this->template write<FragmentStreamA>(std::cout, params_A);
    }

    EXPECT_TRUE(passed_B) << "In: " << tensor_B_in << "\n, Out:\n" << tensor_B_out;
    if (!passed_B) {
      this->template write<FragmentStreamB>(std::cout, params_B);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemmFragmentStream, half_32x32x16_col_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 32, 32>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_128x64x16_col_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 64, 128>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_256x128x16_col_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    1
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    64,
    2
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    128,
    4
  >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemmFragmentStream, half_32x32x16_col_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 32, 32>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_128x64x16_col_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 64, 128>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_256x128x16_col_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    1
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    64,
    2
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    128,
    4
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    128,
    8
  >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemmFragmentStream, half_32x32x16_row_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 32, 32>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_128x64x16_row_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 64, 128>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_256x128x16_row_col) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    32,
    2
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    64,
    4
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    128,
    8
  >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemmFragmentStream, half_32x32x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 32, 32>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_128x64x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 64, 128>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    2
  >().run();
}

TEST(WmmaGemmFragmentStream, half_256x128x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    2
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    64,
    4
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    128,
    8
  >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemmFragmentStream, half4_32x32x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 32, 32>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    4
  >().run();
}

TEST(WmmaGemmFragmentStream, half4_128x64x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 64, 128>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    4
  >().run();
}

TEST(WmmaGemmFragmentStream, half4_256x128x16_row_row) {

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    32,
    4
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    64,
    4
  >().run();

  TestGemmFragmentStream<
    cutlass::Shape<16, 128, 256>,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    128,
    8
  >().run();
}

#endif
