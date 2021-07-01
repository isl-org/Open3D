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
#include "cutlass/reduction/batched_reduction_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x512_nn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x512_nt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x512_tn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x512_tt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x500_nn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 500;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x500_nt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 500;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x500_tn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 500;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_128x256x500_tt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 500;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_1024x64x4096_nn) {
  const int splits_count = 16;
  const int m = 1024;
  const int n = 64;
  const int k = 4096;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_1024x64x4096_nt) {
  const int splits_count = 16;
  const int m = 1024;
  const int n = 64;
  const int k = 4096;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_1024x64x4096_tn) {
  const int splits_count = 16;
  const int m = 1024;
  const int n = 64;
  const int k = 4096;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SplitK_sgemm_128x128x8_splits16, sgemm_1024x64x4096_tt) {
  const int splits_count = 16;
  const int m = 1024;
  const int n = 64;
  const int k = 4096;

  /*batched sgemm traits*/
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<float,
    float,
    float,
    float,
    float, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<SgemmTraits, BatchedReductionTraits>(m, n, k, 1/*partitionK_multiple*/, 2.0f, 1.0f);
}
