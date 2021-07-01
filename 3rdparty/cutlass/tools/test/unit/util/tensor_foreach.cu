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
/* \file

  \brief

    These tests are intended to demonstrate the CUTLASS reference implementation for basic for-each
    operators on the index space of TensorView objects. They instantiate a HostMatrix, initialize
    its elements with random data according to specified random distributions, and clamp the
    elements using a TensorForEach() operation.

    Both device-side and host-side reference implementations are called.
*/

#include "cutlass_unit_test.h"

#include "cutlass/matrix_traits.h"

#include "tools/util/tensor_view_io.h"
#include "tools/util/host_tensor.h"
#include "tools/util/host_matrix.h"

#include "tools/util/reference/device/tensor_foreach.h"
#include "tools/util/reference/device/tensor_elementwise.h"

#include "tools/util/reference/host/tensor_foreach.h"
#include "tools/util/reference/host/tensor_elementwise.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {

/// Define a functor that computes the ReLu operation on a tensor.
template <typename View>
struct ReLuFunc {

  /// Coordinate of index space
  typedef typename View::TensorCoord TensorCoord;

  /// Scalar type
  typedef typename View::Storage T;

  //
  // Data members
  //

  /// Tensor view
  View view;

  /// ReLu threshold
  T threshold;

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  ReLuFunc(View const &view, T threshold): view(view), threshold(threshold) { }

  /// ReLu function
  CUTLASS_HOST_DEVICE
  void operator()(TensorCoord const &coord) {
    T value = view.at(coord);

    if (value < threshold) {
      value = threshold;
    }

    view.at(coord) = value;
  }
};

} // namespace test

///////////////////////////////////////////////////////////////////////////////////////////////////

/// This tests models the computation of ReLu using reference utility code.
TEST(TensorForEach, ReLu_device) {

  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;
  typedef typename HostMatrix::DeviceTensorView View;

  // Define the problem size
  int const M = 517;
  int const N = 117;

  float threshold = 0;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_uniform(-16, 16);

  // RNG seed is hard-coded for determinism in the test.
  int64_t seed = 2080;

  cutlass::reference::device::TensorInitialize(source.device_view(), seed, dist);

  // Define a functor called by TensorForEach<>
  typedef test::ReLuFunc<View> ReLuFunc;

  // Instantiate on host with TensorView and threshold value
  ReLuFunc relu_func(source.device_view(), threshold);

  // Launch kernel that applies the element-wise operator over the tensor's index space.
  cutlass::reference::device::TensorForEach<
    ReLuFunc,
    View::kRank,
    ReLuFunc>(source.size(), relu_func);

  // Verify no element is less than the ReLu threshold.
  source.sync_host();

  int errors = 0;
  for (cutlass::MatrixCoord coord(0, 0); coord.row() < M; ++coord.row()) {
    for (coord.column() = 0; coord.column() < N; ++coord.column()) {
      if (source.at(coord) < threshold) {
        ++errors;
        if (errors < 10) {
          std::cout << "Error - source(" << coord << ") = "
            << source.at(coord) << " is less than threshold " << threshold << std::endl;
        }
      }
    }
  }

  EXPECT_EQ(errors, 0)
    << "Result: " << source;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test to apply the ReLu operation using host-side utilities
TEST(TensorForEach, ReLu_host) {

  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;
  typedef typename HostMatrix::HostTensorView View;

  // Define the problem size
  int const M = 517;
  int const N = 117;

  float threshold = 0;

  bool const kDeviceBacked = false;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor, kDeviceBacked);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_gaussian(-1, 4);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::host::TensorInitialize(source.host_view(), seed, dist);

  // Define a functor called by TensorForEach<>
  typedef test::ReLuFunc<View> ReLuFunc;

  // Instantiate on host with TensorView and threshold value
  ReLuFunc relu_func(source.host_view(), threshold);

  // Invoke host-side for-each computation on the tensor
  cutlass::reference::host::TensorForEach<
    ReLuFunc,
    View::kRank,
    ReLuFunc>(source.size(), relu_func);

  int errors = 0;
  for (cutlass::MatrixCoord coord(0, 0); coord.row() < M; ++coord.row()) {
    for (coord.column() = 0; coord.column() < N; ++coord.column()) {
      if (source.at(coord) < threshold) {
        ++errors;
        if (errors < 10) {
          std::cout << "Error - source(" << coord << ") = "
            << source.at(coord) << " is less than threshold " << threshold << std::endl;
        }
      }
    }
  }

  EXPECT_EQ(errors, 0)
    << "Result: " << source;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
