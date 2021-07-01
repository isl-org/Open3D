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

#include "cutlass/zip_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

/// Kernel which can use tile iterators and zip iterators
template <typename LoadIterator, typename StoreIterator>
__global__ void zip_iterator_kernel(
  typename LoadIterator::Params load_params,
  typename StoreIterator::Params store_params) {

  LoadIterator load_iterator(load_params);
  StoreIterator store_iterator(store_params);

  typename LoadIterator::Fragment fragment;

  load_iterator.load_post_increment(fragment);
  store_iterator.store_post_increment(fragment);
}

} // namespace test

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Test framework
template <typename Scalar, typename Shape>
struct ZipIteratorTest {

  //
  // Type definitions
  //

  static int const kThreadCount = 128;

  typedef cutlass::TileTraitsStandard<Shape, kThreadCount> TileTraits;

  typedef cutlass::TileLoadIterator<TileTraits, Scalar> ScalarLoadIterator;
  typedef cutlass::TileStoreIterator<TileTraits, Scalar> ScalarStoreIterator;

  typedef cutlass::ZipTileIterator<ScalarLoadIterator, ScalarLoadIterator> ZipLoadIterator;
  typedef cutlass::ZipTileIterator<ScalarStoreIterator, ScalarStoreIterator> ZipStoreIterator;

  //
  // Data members
  //

  cutlass::HostMatrix<Scalar> tensor_source_real;
  cutlass::HostMatrix<Scalar> tensor_source_imag;

  cutlass::HostMatrix<Scalar> tensor_dest_real;
  cutlass::HostMatrix<Scalar> tensor_dest_imag;

  //
  // Methods
  //

  /// Ctor
  ZipIteratorTest() {

    tensor_source_real.resize(cutlass::make_Coord(Shape::kH, Shape::kW), cutlass::MatrixLayout::kRowMajor);
    tensor_source_imag.resize(cutlass::make_Coord(Shape::kH, Shape::kW), cutlass::MatrixLayout::kRowMajor);
    tensor_dest_real.resize(cutlass::make_Coord(Shape::kH, Shape::kW), cutlass::MatrixLayout::kRowMajor);
    tensor_dest_imag.resize(cutlass::make_Coord(Shape::kH, Shape::kW), cutlass::MatrixLayout::kRowMajor);
  }

  /// Runs test
  void run() {

    tensor_source_real.fill_sequential();
    tensor_source_imag.fill_sequential();

    tensor_dest_real.fill(0);
    tensor_dest_imag.fill(0);

    tensor_source_real.sync_device();
    tensor_source_imag.sync_device();
    tensor_dest_real.sync_device();
    tensor_dest_imag.sync_device();


    typename ZipLoadIterator::Params load_params;
    typename ZipStoreIterator::Params store_params;

    load_params.first.initialize(
      tensor_source_real.device_data(),
      0,
      tensor_source_real.leading_dim(),
      1
    );

    load_params.second.initialize(
      tensor_source_imag.device_data(),
      0,
      tensor_source_real.leading_dim(),
      1
    );

    store_params.first.initialize(
      tensor_dest_real.device_data(),
      0,
      tensor_source_real.leading_dim(),
      1
    );

    store_params.second.initialize(
      tensor_dest_imag.device_data(),
      0,
      tensor_source_real.leading_dim(),
      1
    );

    /// Launch kernel
    test::zip_iterator_kernel<ZipLoadIterator, ZipStoreIterator><<<
      dim3(1,1),
      dim3(kThreadCount, 1)
    >>>(
      load_params,
      store_params
    );

    cudaError_t result = cudaGetLastError();
    EXPECT_EQ(result, cudaSuccess) << "Error on kernel launch: " << cudaGetErrorString(result);

    tensor_dest_real.sync_host();
    tensor_dest_imag.sync_host();

    // Verify equivalence
    EXPECT_TRUE(tensor_dest_real.bit_equals(tensor_source_real));
    EXPECT_TRUE(tensor_dest_imag.bit_equals(tensor_source_imag));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ZipTileIterator, tile_128x8) {
  ZipIteratorTest<int, cutlass::Shape<1, 8, 128> >().run();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

