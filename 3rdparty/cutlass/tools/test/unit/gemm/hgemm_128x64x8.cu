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
#include "cutlass/gemm/hgemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x1_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x8_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x9_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x16_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x64_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x64x16_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x128x16_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x128x16_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x2_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x8_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x10_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x16_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x64_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x64x16_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x128x16_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x128x16_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(Hgemm_128x64x8, hgemm_128x64x8_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x10_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x16_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x64_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x64x16_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x128x16_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x128x16_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x8_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x10_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x16_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x64x64_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x64x16_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_128x128x16_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(128, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_128x64x8, hgemm_256x128x16_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 64, 128> >
      HgemmTraits;
  run_gemm<HgemmTraits>(256, 128, 16);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

