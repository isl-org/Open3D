/***************************************************************************************************
* Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/cutlass.h"

template <typename GemmTraits_>
static void run_binary_gemm(int m, int n, int k, int alpha = 1, int beta = 1) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  typename Gemm::Params params;

  test::GemmTestbed<cutlass::Vector<cutlass::bin1_t, 32>,  // AType
                    cutlass::Vector<cutlass::bin1_t, 32>,  // BType
                    int32_t,                               // CType
                    int32_t,                               // Accumulator
                    int                                    // Scalar
                    >
      testbed(m,
              n,
              k / 32,
              test::convert(GemmTraits_::kLayoutA),
              test::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  // Initializes the input vectors for computation
  testbed.initialize_binary();

  // Compute the reference result on the host (CPU)
  testbed.compute_host();

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K() * 32,
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

  testbed.computed.sync_host();

  // Check the results
  ASSERT_TRUE(testbed.computed.bit_equals(testbed.ref_host));
}
