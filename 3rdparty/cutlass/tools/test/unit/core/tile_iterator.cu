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
#include "cutlass_unit_test.h"
#include "tools/util/host_matrix.h"
#include "tools/util/tensor_view_io.h"
#include "cutlass/shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tile_iterator.h"
#include "cutlass/tile_traits_standard.h"
#include "cutlass/iterator_access.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

template <typename Traits, typename Scalar>
__global__ void load_store_global(
            typename cutlass::TileLoadIterator<Traits, Scalar, cutlass::IteratorAdvance::kH,
            cutlass::MemorySpace::kGlobal>::Scalar const *input,
            typename cutlass::TileStoreIterator<Traits, Scalar, cutlass::IteratorAdvance::kH,
            cutlass::MemorySpace::kGlobal>::Scalar *output,
            int kW,
            int kH
        ) {

    /// Load iterator
    typedef cutlass::TileLoadIterator<Traits, Scalar, cutlass::IteratorAdvance::kH, cutlass::MemorySpace::kGlobal> LoadIterator;
    /// Store iterator
    typedef cutlass::TileStoreIterator<Traits, Scalar, cutlass::IteratorAdvance::kH, cutlass::MemorySpace::kGlobal> StoreIterator;
    /// Predicate vector
    typedef typename LoadIterator::PredicateVector PredicateVector;

    typename LoadIterator::Params load_params;
    typename StoreIterator::Params store_params;

    typedef typename Traits::Tile Tile;

    load_params.initialize(input, Tile::kH*Tile::kW, Tile::kW, 1);
    store_params.initialize(output, Tile::kH*Tile::kW, Tile::kW, 1);

    LoadIterator load_iterator(load_params);
    StoreIterator store_iterator(store_params);
    PredicateVector predicates;

    load_iterator.initialize_predicates(predicates.begin(), cutlass::make_Coord(1, kH, kW));

    typename LoadIterator::Fragment fragment;

    load_iterator.load_post_increment(fragment, predicates.begin());
    store_iterator.store_post_increment(fragment);
}

/// Launches the load_store_global test
template <typename Scalar, typename Tile, int kThreadsPerThreadBlock>
void run_load_store_global(int kW, int kH) {

  typedef cutlass::TileTraitsStandard<Tile, kThreadsPerThreadBlock> Traits;

  typedef typename cutlass::TypeTraits<Scalar>::device_type ScalarDevice;

  cutlass::HostMatrix<Scalar> input;
  cutlass::HostMatrix<Scalar> output;

  input.resize(cutlass::make_Coord(Tile::kW, Tile::kH), cutlass::MatrixLayout::kColumnMajor);
  output.resize(cutlass::make_Coord(Tile::kW, Tile::kH), cutlass::MatrixLayout::kColumnMajor);

  input.fill_linear(cutlass::make_Coord(1, Tile::kW));
  output.fill(0);

  test::load_store_global<Traits, ScalarDevice> <<<
      dim3(1, 1, 1),
      dim3(kThreadsPerThreadBlock, 1)
    >>>(input.device_data(), output.device_data(), kW, kH);

  cudaError_t result = cudaDeviceSynchronize();

  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                               << "\n";
  output.sync_host();

  bool passed = true;
  for(int i = 0; i < Tile::kW; ++i) {
    for(int j = 0; j < Tile::kH; ++j) {
      if(i < kW && j < kH && output.at(cutlass::make_Coord(i, j)) != Scalar(Tile::kW*j+i)){
        std::cout << "FAILED: (" << i << ", " << j
                  << ") -- expected: " << (Tile::kW*j+i)
                  << ", actual: " << output.at(cutlass::make_Coord(i, j))
                  << std::endl;
        passed = false;
        break;
      }
    }
  }

  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_128x8_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(128, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_128x8_rake) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 32>(128, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_127x8_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(127, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_129x8_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(129, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_112x8_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(112, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_67x8_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(67, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_113x7_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(113, 7);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_113x10_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(113, 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_131x7_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(131, 7);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_131x9_contiguous) {
    run_load_store_global<float, cutlass::Shape<1, 8, 128>, 128>(131, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Half
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_128x8_contiguous_f16) {
    run_load_store_global<cutlass::half_t, cutlass::Shape<1, 8, 128>, 128>(128, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Double
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_128x8_contiguous_f64) {
    run_load_store_global<double, cutlass::Shape<1, 8, 128>, 128>(128, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Int
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TileIterator, tile_128x8_contiguous_s32) {
    run_load_store_global<int, cutlass::Shape<1, 8, 128>, 128>(128, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace test
