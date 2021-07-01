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
#include "cutlass/gemm/sgemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

 ////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_1024x512x8_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  run_gemm<SgemmTraits>(1024, 512, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x81x1_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 81, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x8_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x9_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x73x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 73, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_97x112x64_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(97, 112, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x112x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x240x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 240, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x240x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 240, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x1_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_79x112x8_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(79, 112, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x81x9_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 81, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x73x64_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 73, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x112x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x256x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x256x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x128x1_tn) {
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> > SgemmTraits;
    run_gemm<SgemmTraits>(128, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_127x112x8_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(127, 112, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_21x112x9_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(21, 112, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x73x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 73, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x81x64_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 81, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x112x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_47x256x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(47, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_211x256x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(211, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x128x1_tt) {
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> > SgemmTraits;
    run_gemm<SgemmTraits>(128, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_109x112x8_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(109, 112, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x9_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 9);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_123x112x64_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(123, 112, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x112x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 112, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x256x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_256x256x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_120x112x64_ldg4_nt) {
  // Load 4 floats per LDG for A/B.
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<8, 128, 128>,
                                     cutlass::gemm::LinearScaling<float>,
                                     cutlass::Shape<8, 8, 8>,
                                     4, 4>
      SgemmTraits;
  run_gemm<SgemmTraits>(120, 112, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x128x16_alpha2_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 128, 16, 2.f, 0.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x16_beta1_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 16, 1.f, 1.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x128x8, sgemm_128x112x16_alpha2_beta1_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 112, 16, 2.f, 1.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
