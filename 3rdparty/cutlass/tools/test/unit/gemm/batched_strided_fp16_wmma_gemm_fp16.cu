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
#include "cutlass/wmma_matrix.h"
#if defined(CUTLASS_USE_WMMA_API)

#include "cutlass_unit_test.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_32x32x16_nn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    8, /*kScalarsPerLdgA_*/
    8, /*kScalarsPerLdgB_*/
    8, /*KScalarsPerLdsA_*/
    8, /*KScalarsPerLdsB_*/
    16 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    16 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    16 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(32, 32, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_32x32x16_nt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    8, /*kScalarsPerLdgA_*/
    8, /*kScalarsPerLdgB_*/
    8, /*KScalarsPerLdsA_*/
    8, /*KScalarsPerLdsB_*/
    16 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    16 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    16 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(32, 32, 64, 3);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_32x32x16_tn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    8, /*kScalarsPerLdgA_*/
    8, /*kScalarsPerLdgB_*/
    8, /*KScalarsPerLdsA_*/
    8, /*KScalarsPerLdsB_*/
    16 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    16 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    16 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(32, 32, 64, 3);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_32x32x16_tt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    8, /*kScalarsPerLdgA_*/
    8, /*kScalarsPerLdgB_*/
    8, /*KScalarsPerLdsA_*/
    8, /*KScalarsPerLdsB_*/
    16 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    16 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    16 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(32, 32, 64, 3);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//mulitple of 4
TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_36x36x16_nn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    4, /*kScalarsPerLdgA_*/
    4, /*kScalarsPerLdgB_*/
    4, /*KScalarsPerLdsA_*/
    4, /*KScalarsPerLdsB_*/
    8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(36, 36, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_36x36x16_nt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    4, /*kScalarsPerLdgA_*/
    4, /*kScalarsPerLdgB_*/
    4, /*KScalarsPerLdsA_*/
    4, /*KScalarsPerLdsB_*/
    8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(36, 36, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_36x36x16_tn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    4, /*kScalarsPerLdgA_*/
    4, /*kScalarsPerLdgB_*/
    4, /*KScalarsPerLdsA_*/
    4, /*KScalarsPerLdsB_*/
    8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(36, 36, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_36x36x16_tt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    4, /*kScalarsPerLdgA_*/
    4, /*kScalarsPerLdgB_*/
    4, /*KScalarsPerLdsA_*/
    4, /*KScalarsPerLdsB_*/
    8 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    8 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    8 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(36, 36, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//mulitple of 2
TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_34x34x16_nn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    2, /*kScalarsPerLdgA_*/
    2, /*kScalarsPerLdgB_*/
    2, /*KScalarsPerLdsA_*/
    2, /*KScalarsPerLdsB_*/
    4 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    4 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    4 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(34, 34, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_34x34x16_nt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    2, /*kScalarsPerLdgA_*/
    2, /*kScalarsPerLdgB_*/
    2, /*KScalarsPerLdsA_*/
    2, /*KScalarsPerLdsB_*/
    4 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    4 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    4 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(34, 34, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_34x34x16_tn) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    2, /*kScalarsPerLdgA_*/
    2, /*kScalarsPerLdgB_*/
    2, /*KScalarsPerLdsA_*/
    2, /*KScalarsPerLdsB_*/
    4 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    4 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    4 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(34, 34, 64, 3);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_strided_batched_16x16x32_f32, fp16_wmma_gemm_fp16_34x34x16_tt) {
  typedef float accumu_type;
  typedef half c_type;
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    c_type,
    cutlass::gemm::LinearScaling<accumu_type>,
    accumu_type,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    2, /*kScalarsPerLdgA_*/
    2, /*kScalarsPerLdgB_*/
    2, /*KScalarsPerLdsA_*/
    2, /*KScalarsPerLdsB_*/
    4 / sizeof(c_type), /*kScalarsPerLdgCAndStgD_*/
    4 / sizeof(accumu_type), /*kScalarsPerStsD_*/
    4 / sizeof(accumu_type)  /*kScalarsPerLdsD_*/
  >
    WmmaGemmTraits;

  run_batched_strided_gemm<WmmaGemmTraits>(34, 34, 64, 3);
}

#endif
