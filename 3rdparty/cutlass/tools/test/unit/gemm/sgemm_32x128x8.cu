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

TEST(Sgemm_32x128x8, sgemm_32x128x1_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
                                     cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
      SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x8_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 8);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x256x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_64x256x16_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(64, 256, 16);
}

//NN
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x1_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x8_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 8);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x256x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_64x256x16_nn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(64, 256, 16);
}

//TN
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x1_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x8_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 8);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x256x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_64x256x16_tn) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(64, 256, 16);
}

//TT
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x1_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x8_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 8);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x128x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 128, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_32x256x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(32, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_32x128x8, sgemm_64x256x16_tt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 32>,
    cutlass::gemm::LinearScaling<float>, cutlass::Shape<8, 8, 4> >
    SgemmTraits;
  run_gemm<SgemmTraits>(64, 256, 16);
}
