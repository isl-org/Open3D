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

    These tests initialize host- and device-side tensors according to several random distributions.
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

#define ENABLE_OUTPUT 0 // Supress output by default.

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorInitialize, uniform_device) {

  // Define the problem size
  int const M = 517;
  int const N = 117;

  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_uniform(0, 128, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::device::TensorInitialize(source.device_view(), seed, dist);

  source.sync_host();

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_uniform_device.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

TEST(TensorInitialize, uniform_host) {

  // Define the problem size
  int const M = 517;
  int const N = 117;

  bool const kDeviceBacked = false;

  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor, kDeviceBacked);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_uniform(0, 128, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::host::TensorInitialize(source.host_view(), seed, dist);

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_uniform_host.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

TEST(TensorInitialize, gaussian_device) {

  // Define the problem size
  int const M = 517;
  int const N = 117;


  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_gaussian(1, 2, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::device::TensorInitialize(source.device_view(), seed, dist);

  source.sync_host();

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_gaussian_device.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

TEST(TensorInitialize, gaussian_host) {
  // Define the problem size
  int const M = 517;
  int const N = 117;

  bool const kDeviceBacked = false;

  // Define HostMatrix type
  typedef cutlass::HostMatrix<float> HostMatrix;

  // Construct the host matrix
  HostMatrix source(cutlass::MatrixCoord(M, N), cutlass::MatrixLayout::kRowMajor, kDeviceBacked);
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_gaussian(1, 2, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::host::TensorInitialize(source.host_view(), seed, dist);

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_gaussian_host.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Interleaved matrix layouts
//
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorInitialize, interleaved_gaussian_device) {

  // Define the problem size
  int const M = 512;
  int const N = 128;

  // Define a mapping function for column-major interleaved layout
  int const kInterleave = 4;
  typedef cutlass::MatrixLayout::ColumnMajorInterleaved<kInterleave> TensorRefMapFunc;

  // Construct a rank=2 host tensor of size M-by-N
  cutlass::HostTensor<
    float,
    2,
    TensorRefMapFunc > source(TensorRefMapFunc::stride(M), cutlass::make_Coord(M, N));

  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_gaussian(1, 2, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::device::TensorInitialize(source.device_view(), seed, dist);

  source.sync_host();

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_interleaved_gaussian_device.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

TEST(TensorInitialize, interleaved_gaussian_host) {
  // Define the problem size
  int const M = 512;
  int const N = 128;

  bool const kDeviceBacked = false;

  // Define a mapping function for column-major interleaved layout
  int const kInterleave = 4;
  typedef cutlass::MatrixLayout::ColumnMajorInterleaved<kInterleave> TensorRefMapFunc;

  // Construct a rank=2 host tensor of size M-by-N
  cutlass::HostTensor<
    float,
    2,
    TensorRefMapFunc > source(TensorRefMapFunc::stride(M), cutlass::make_Coord(M, N), kDeviceBacked);

  // Construct the host matrix
  source.fill(0);

  // Initialize the source matrix with a uniform distribution
  cutlass::Distribution dist;
  dist.set_gaussian(1, 2, -1);

  // RNG seed is hard-coded for determinism in the test.
  unsigned seed = 2080;

  cutlass::reference::host::TensorInitialize(source.host_view(), seed, dist);

  if (ENABLE_OUTPUT) {
    std::ofstream result("TensorInitialize_interleaved_gaussian_host.csv");

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        result << source.at(cutlass::make_Coord(i, j)) << "\n";
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Comparison operator
//
///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TensorEquals, interleaved_device) {

  // Define the problem size
  int const M = 512;
  int const N = 128;

  // Define a mapping function for column-major interleaved layout
  int const kInterleave = 4;
  typedef cutlass::MatrixLayout::ColumnMajorInterleaved<kInterleave> TensorRefMapFunc;

  // Construct two rank=2 host tensor of size M-by-N
  cutlass::HostTensor<
    float,
    2,
    TensorRefMapFunc > left(TensorRefMapFunc::stride(M), cutlass::make_Coord(M, N));

  cutlass::HostTensor<
    float,
    2,
    TensorRefMapFunc > right(TensorRefMapFunc::stride(M), cutlass::make_Coord(M, N));

  // Initialize
  left.fill_sequential();
  right.fill_sequential();

  // Assert equality
  EXPECT_TRUE(cutlass::reference::device::TensorEquals(left.device_view(), right.device_view()));

  // Overwrite one with an unexpected element
  left.at(cutlass::make_Coord(24, 17)) = -1;
  left.sync_device();

  // Assert inequality
  EXPECT_FALSE(cutlass::reference::device::TensorEquals(left.device_view(), right.device_view()));
}

TEST(TensorEquals, interleaved_host) {

}

///////////////////////////////////////////////////////////////////////////////////////////////////
