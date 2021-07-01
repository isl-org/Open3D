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

#pragma once

#include <utility>
#include "cutlass/cutlass.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "cutlass/gemm/device_gemm.h"
#include "cutlass/gemm/device_gemm_traits.h"

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1.0f),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0.0f)) {

  //typedef typename GemmTraits_::KernelClass Gemm;
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;

  typename Gemm::Params params;

  test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      >
      testbed(m,
              n,
              k,
              lda,
              ldb,
              ldc,
              test::convert(GemmTraits_::kLayoutA),
              test::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

  if (testbed.has_cublas_support()) {
    EXPECT_TRUE(testbed.verify_host_with_cublas());
    EXPECT_TRUE(testbed.verify_reference_with_cublas());
  }


  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
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

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    ASSERT_TRUE(testbed.verify_with_host());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1.0f),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0.0f)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  //typedef typename GemmTraits_::KernelClass Gemm;
  typename Gemm::Params params;

  typedef test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      > GemmTestbed;

  GemmTestbed testbed(m,
              n,
              k,
              test::convert(GemmTraits_::kLayoutA),
              test::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

  if (testbed.has_cublas_support()) {
    EXPECT_TRUE(testbed.verify_host_with_cublas());
    EXPECT_TRUE(testbed.verify_reference_with_cublas());
  }

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
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

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    ASSERT_TRUE(testbed.verify_with_host());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_batched_strided_gemm(
    int m,
    int n,
    int k,
    int batch_count,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1.0f),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0.0f)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  //typedef typename GemmTraits_::KernelClass Gemm;
  typename Gemm::Params params;
  test::GemmTestbed<
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
      typename test::GemmTestbedTraits<
          typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
      typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
      >
      testbed(m,
              n,
              k,
              batch_count,
              test::convert(GemmTraits_::kLayoutA),
              test::convert(GemmTraits_::kLayoutB),
              alpha,
              beta);

  testbed.initialize();

  // host support is not implemented for strided batched gemm
  // if (testbed.has_cublas_support()) {
  //  EXPECT_TRUE(testbed.verify_host_with_cublas());
  //}

  params.initialize(testbed.M(),
                    testbed.N(),
                    testbed.K(),
                    testbed.alpha,
                    testbed.ptr_A(),
                    testbed.lda(),
                    testbed.get_batch_stride_A(),
                    testbed.ptr_B(),
                    testbed.ldb(),
                    testbed.get_batch_stride_B(),
                    testbed.beta,
                    testbed.ptr_C_initial(),
                    testbed.ldc(),
                    testbed.get_batch_stride_C(),
                    testbed.ptr_computed(),
                    testbed.ldc(),
                    testbed.get_batch_stride_C(),
                    testbed.get_batch_count());

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
                                 << "\n";

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  } else {
    // ASSERT_TRUE(testbed.verify_with_host());
    ASSERT_TRUE(false) << "host support is not implemented for strided batched gemm" << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_, typename ReductionTraits_>
static void run_splitK_gemm(int m,
  int n,
  int k,
  int partitionK_multiple = 1, /*requires each partition to be mulitple of partitionK_multiple*/
  typename test::GemmTestbedTraits<typename ReductionTraits_::ScalarAlphaBeta>::host_type alpha =
      typename test::GemmTestbedTraits<typename ReductionTraits_::ScalarAlphaBeta>::host_type(1.0f),
  typename test::GemmTestbedTraits<typename ReductionTraits_::ScalarAlphaBeta>::host_type beta =
      typename test::GemmTestbedTraits<typename ReductionTraits_::ScalarAlphaBeta>::host_type(0.0f),
  bool use_host_reference = false){

  test::GemmTestbed<
    typename test::GemmTestbedTraits<
      typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
    typename test::GemmTestbedTraits<
      typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
    typename test::GemmTestbedTraits<
      typename ReductionTraits_::ScalarC>::host_type,  // CType
    typename test::GemmTestbedTraits<
      typename GemmTraits_::GemmConfig::ScalarD>::host_type,  // Workspace Accumulator
    typename test::GemmTestbedTraits<typename ReductionTraits_::ScalarAlphaBeta>::host_type  // Scalar
  >
    testbed(m,
      n,
      k,
      test::convert(GemmTraits_::kLayoutA),
      test::convert(GemmTraits_::kLayoutB),
      alpha,
      beta);

  testbed.initialize();

  // create a device gemm
  typedef cutlass::gemm::SplitkPIGemmTraits<GemmTraits_, ReductionTraits_> deviceGemmTraits;
  //typedef typename deviceGemmTraits::KernelClass deviceGemm;
  typedef typename cutlass::gemm::DeviceGemm<deviceGemmTraits> deviceGemm;
  typename deviceGemm::Params deviceGemmParams(testbed.M(), testbed.N(), testbed.K());

  // query if workspace is needed
  size_t workspace_size = deviceGemmParams.required_workspace_memory_in_byte();
  typename test::GemmTestbedTraits<typename GemmTraits_::GemmConfig::ScalarD>::device_type
    *workspace_ptr = 0;
  if (workspace_size != 0) {
    cudaError_t workspace_err = cudaMalloc(&workspace_ptr, workspace_size);
    ASSERT_EQ(workspace_err, cudaSuccess) << "\nCUDA workspace malloc error: " << cudaGetErrorString(workspace_err)
      << "\n";
  }

  deviceGemmParams.initialize(testbed.alpha,
    testbed.ptr_A(),
    testbed.lda(),
    testbed.ptr_B(),
    testbed.ldb(),
    testbed.beta,
    testbed.ptr_C_initial(),
    testbed.ldc(),
    testbed.ptr_computed(),
    testbed.ldc(),
    workspace_ptr,
    partitionK_multiple);


  deviceGemm::launch(deviceGemmParams);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
    << "\n";

  if (workspace_size != 0) {
    cudaError_t workspace_err = cudaFree(workspace_ptr);
    ASSERT_EQ(workspace_err, cudaSuccess) << "\nCUDA workspace free error: " << cudaGetErrorString(workspace_err)
      << "\n";
  }

  if (use_host_reference == true || testbed.has_cublas_support() == false) {
    ASSERT_TRUE(testbed.verify_with_host());
  }
  else {
    ASSERT_TRUE(testbed.verify_with_cublas());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_partitioned_k_gemm(
  int m,
  int n,
  int k,
  int partitionK_count,
  int partitionK_multiple = 1, //requires each partition to be multiples of partitionK_multiple
  typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
  typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1.0f),
  typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
  typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0.0f)) {
  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;
  //typedef typename GemmTraits_::KernelClass Gemm;
  typename Gemm::Params params;
  test::GemmTestbed<
    typename test::GemmTestbedTraits<
    typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
    typename test::GemmTestbedTraits<
    typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
    typename test::GemmTestbedTraits<
    typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
    typename test::GemmTestbedTraits<
    typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
  >
    testbed(m,
      n,
      std::make_pair(k, partitionK_count),
      partitionK_multiple,
      test::convert(GemmTraits_::kLayoutA),
      test::convert(GemmTraits_::kLayoutB),
      alpha,
      beta);

  testbed.initialize();

  // host support is not implemented for strided batched gemm
  // if (testbed.has_cublas_support()) {
  //  EXPECT_TRUE(testbed.verify_host_with_cublas());
  //}

  params.initialize(testbed.M(),
    testbed.N(),
    testbed.K(),
    testbed.alpha,
    testbed.ptr_A(),
    testbed.lda(),
    testbed.ptr_B(),
    testbed.ldb(),
    testbed.beta,
    testbed.ptr_C_initial(),
    testbed.ldc(),
    testbed.ptr_computed(),
    testbed.ldc(),
    partitionK_count,
    partitionK_multiple);

  Gemm::launch(params);

  cudaError_t result = cudaDeviceSynchronize();
  ASSERT_EQ(result, cudaSuccess) << "\nCUDA kernel launch error: " << cudaGetErrorString(result)
    << "\n";

  if (testbed.has_cublas_support()) {
    ASSERT_TRUE(testbed.verify_with_cublas());
  }
  else {
    // ASSERT_TRUE(testbed.verify_with_host());
    ASSERT_TRUE(false) << "host support is not implemented for strided batched gemm" << std::endl;
  }
}
