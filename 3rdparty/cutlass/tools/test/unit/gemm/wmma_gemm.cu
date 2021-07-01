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
//
// FP16 accumulation
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32_f16, wmma_gemm_16x16x16_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<half>,
    half
  >
  WmmaGemmTraits;

  run_gemm<WmmaGemmTraits>(16, 16, 16);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32_f16, wmma_gemm_16x16x32_nn) {

  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<half>,
    half
  >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_128x128x32_f16, wmma_16x16x16_gemm_256x256x128_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<half>,
    half
  >
    WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x16_nt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x32_nt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_128x128x32, wmma_16x16x16_gemm_256x256x128_nt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_8x32x16_gemm_256x256x128_nt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 32, 8> >

      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_32x8x16_gemm_256x256x128_nt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 8, 32> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x16_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x32_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_128x128x32, wmma_16x16x16_gemm_256x256x128_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_8x32x16_gemm_256x256x128_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 32, 8> >

      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_32x8x16_gemm_256x256x128_nn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 8, 32> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x16_tt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x32_tt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_128x128x32, wmma_16x16x16_gemm_256x256x128_tt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_8x32x16_gemm_256x256x128_tt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 32, 8> >

      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_32x8x16_gemm_256x256x128_tt) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 8, 32> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x16_tn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_16x16x32, wmma_gemm_16x16x32_tn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 16, 16> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(16, 16, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaGemm_128x128x32, wmma_16x16x16_gemm_256x256x128_tn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_8x32x16_gemm_256x256x128_tn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 32, 8> >

      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_VERSION) && CUDA_VERSION >= 9100
TEST(WmmaGemm_128x128x32, wmma_32x8x16_gemm_256x256x128_tn) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 128, 128>,
                                        half,
                                        half,
                                        float,
                                        cutlass::gemm::LinearScaling<float>,
                                        float,
                                        cutlass::Shape<32, 64, 64>,
                                        cutlass::Shape<16, 8, 32> >
      WmmaGemmTraits;
  run_gemm<WmmaGemmTraits>(256, 256, 128);
}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
#endif  // defined CUTLASS_USE_WMMA_API
