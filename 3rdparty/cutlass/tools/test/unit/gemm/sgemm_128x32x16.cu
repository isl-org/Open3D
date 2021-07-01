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

TEST(Sgemm_128x32x16, sgemm_128x32x1_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x16_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x17_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 17);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x32_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x32x16_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x64x16_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x64x16_nt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x1_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x16_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x17_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 17);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x32_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x32x16_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x64x16_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x64x16_nn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x1_tn) {
    typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<16, 128, 128> > SgemmTraits;
    run_gemm<SgemmTraits>(128, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x16_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x17_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 17);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x32_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x32x16_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x64x16_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x64x16_tn) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x1_tt) {
    typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 128, 128> > SgemmTraits;
    run_gemm<SgemmTraits>(128, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x16_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x17_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 17);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x32x32_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x32x16_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 32, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_128x64x16_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(128, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_128x32x16, sgemm_256x64x16_tt) {
  typedef cutlass::gemm::SgemmLBTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<16, 32, 128> >
      SgemmTraits;
  run_gemm<SgemmTraits>(256, 64, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
