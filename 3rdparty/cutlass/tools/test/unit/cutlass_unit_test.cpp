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
/** \file
    \brief CUTLASS Unit Tests
*/

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

/// Sets flags for Unit test
void set_gtest_flag() {
  // Default flags can be overwritten by --gtest_filter from commandline
  cudaError_t err;

  int cudaDeviceId;
  err = cudaGetDevice(&cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Error: Could not detect active GPU device ID"
              << " [" << cudaGetErrorString(err) << "]" << std::endl;
    exit(1);
  }

  cudaDeviceProp deviceProperties;
  err = cudaGetDeviceProperties(&deviceProperties, cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Error: Could not get device properties for GPU " << cudaDeviceId << " ["
              << cudaGetErrorString(err) << "]" << std::endl;
    exit(1);
  }

  int deviceMajorMinor = deviceProperties.major * 10 + deviceProperties.minor;

  // Defines text filters for each GEMM kernel based on minimum supported compute capability
  struct {

    /// Unit test filter string
    char const *filter;

    /// Minimum compute capability for the kernels in the named test
    int compute_capability;

    /// If true, the tests are enabled strictly for one compute capability
    bool experimental;
  } test_filters[] = {
    { "Sgemm*",                     50, false },
    { "*sgemm*",                    50, false },
    { "Dgemm*",                     60, false },
    { "*dgemm*",                    60, false },
    { "Fp16_sgemm*",                60, false },
    { "*fp16_sgemm*",               60, false },
    { "Batched_reduction*",         60, false },
    { "*batched_reduction*",        60, false },
    { "Float_batched_reduction*",   60, false },
    { "*float_batched_reduction*",  60, false },
    { "SplitK*",                    60, false },
    { "*splitK*",                   60, false },
    { "Hgemm*",                     60, false },
    { "*hgemm*",                    60, false },
    { "Igemm*",                     61, false },
    { "*igemm*",                    61, false },
    { "WmmaGemm*",                  70, false },
    { "*wmma*",                     70, false },
    { "WmmaInt8*",                  72, false },
    { "*wmmaInt8*",                 72, false },
    { "WmmaInt4*",                  75, true },
    { "*wmmaInt4*",                 75, true },
    { "WmmaBinary*",                75, true },
    { "*wmmaBinary*",               75, true },
    { 0, 0, false }
  };

  // Set negative test filters
  std::stringstream ss;
  ss << "-";
  for (int i = 0, j = 0; test_filters[i].filter; ++i) {
    if (deviceMajorMinor < test_filters[i].compute_capability ||
        (test_filters[i].experimental && deviceMajorMinor != test_filters[i].compute_capability)) {

      ss << (j++ ? ":" : "") << test_filters[i].filter;
    }
  }

  ::testing::GTEST_FLAG(filter) = ss.str();
}

int main(int argc, char* arg[]) {
  set_gtest_flag();
  ::testing::InitGoogleTest(&argc, arg);
  return RUN_ALL_TESTS();
}
