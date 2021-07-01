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
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/linear_scaling_device_ptr.h"
#include "cutlass/gemm/sgemm_traits.h"

#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// This example defines an SGEMM with a linear scaling functor that supports optionally passing
// alpha and beta via device-side pointers as in cuBLAS.
TEST(Sgemm_epilogue_functor, device_ptr_mode_sgemm_1024x512x128_nt) {

  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<8, 128, 128>,
    cutlass::gemm::LinearScalingDevicePtr<float>
  >
    SgemmTraits;

  // Define a GEMM problem size
  int const m = 1025;
  int const n = 512;
  int const k = 128;

  // Define scalars
  float alpha_host = 3;
  float beta_host = 2;

  // Define a device-backed tensor to contain the scalars
  cutlass::HostTensor<float, 1> device_scalars(2);

  // Copy scalar values to device memory for device-ptr mode
  device_scalars.at(0) = alpha_host;
  device_scalars.at(1) = beta_host;
  device_scalars.sync_device();

  // Construct a GemmTestbed instance
  test::GemmTestbed<
    float,  // AType
    float,  // BType
    float,  // CType
    float,  // Accumulator
    float   // Scalar
    >
    testbed(m,
            n,
            k,
            test::convert(SgemmTraits::kLayoutA),
            test::convert(SgemmTraits::kLayoutB),
            alpha_host,
            beta_host);

  testbed.initialize();

  //
  // Construct a CUTLASS GEMM and initialize parameters
  //
  typedef cutlass::gemm::Gemm<SgemmTraits> Gemm;
  typename Gemm::Params params;

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    0,                // alpha ignored
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    0,                // beta ignored
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.ptr_computed(),
                    testbed.ldc());

  // Explicitly call the epilogue functor's initialize method to pass additional arguments
  params.epilogue.functor.initialize(
    device_scalars.device_data() + 0,   // pointer to alpha in device memory
    device_scalars.device_data() + 1);  // pointer to beta in device memory

  // Launch the CUTLASS SGEMM kernel
  Gemm::launch(params);

  // Report any errors
  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess)
    << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
    << "\n";

  // Verify result
  ASSERT_TRUE(testbed.verify_with_cublas());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
