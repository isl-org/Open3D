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
#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_SUBBYTE_WMMA

#include "cutlass_unit_test.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/binary_gemm.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_32x32x256, wmma_binary_gemm_32x32x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(32, 32, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_32x32x512, wmma_binary_gemm_32x32x512) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<512, 32, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<512, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(32, 32, 512);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_64x64x256, wmma_binary_gemm_64x64x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 64, 64>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(64, 64, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_64x32x256, wmma_binary_gemm_64x32x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 32, 64>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(64, 32, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_32x64x256, wmma_binary_gemm_32x64x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 64, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(32, 64, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_128x128x256, wmma_binary_gemm_128x128x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 128, 128>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 64, 64>,
                                        cutlass::Shape<128, 8, 8>,
                                        128,
                                        128>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(128, 128, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_128x128x256, wmma_binary_gemm_512x512x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 128, 128>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 64, 64>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(512, 512, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_32x32x256, wmma_binary_gemm_32x32x512) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(32, 32, 512);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_32x32x1024, wmma_binary_gemm_128x128x1024) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<1024, 128, 128>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<1024, 32, 32>,
                                        cutlass::Shape<128, 8, 8>,
                                        128,
                                        128>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(128, 128, 1024);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaBinaryGemm_64x32x1024, wmma_binary_gemm_128x128x1024) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<1024, 128, 128>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        cutlass::Vector<cutlass::bin1_t, 32>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<1024, 32, 64>,
                                        cutlass::Shape<128, 8, 8>,
                                        128,
                                        128>
      WmmaGemmTraits;
  run_binary_gemm<WmmaGemmTraits>(128, 128, 1024);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // ifdef CUTLASS_USE_SUBBYTE_WMMA
