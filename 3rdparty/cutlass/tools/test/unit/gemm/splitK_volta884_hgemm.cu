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
#include <cublas_v2.h>
#include <cstring>
#include "cutlass_unit_test.h"

#include "tools/util/half.h"
#include "tools/util/host_tensor.h"
#include "tools/util/tensor_view_io.h"

#include "cutlass/gemm/volta884_gemm_traits.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/reduction/batched_reduction_traits.h"

#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

#if CUTLASS_ENABLE_TENSOR_CORE_MMA


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits16, volta884_h884gemm_128x256x512_nn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits16, volta884_h884gemm_128x256x512_nt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits16, volta884_h884gemm_128x256x512_tn) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits16, volta884_h884gemm_128x256x512_tt) {
  const int splits_count = 16;
  const int m = 128;
  const int n = 256;
  const int k = 512;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x88_nn) {
  /*
  m = 128, n = 256, overall_K = 88, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 88;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x88_nt) {
  /*
  m = 128, n = 256, overall_K = 88, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 88;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x88_tn) {
  /*
  m = 128, n = 256, overall_K = 88, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 88;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x88_tt) {
  /*
  m = 128, n = 256, overall_K = 88, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 88;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x256_nn) {
  /*
  m = 128, n = 256, overall_K = 256, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 25
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 24
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 40
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 256;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x256_nt) {
  /*
  m = 128, n = 256, overall_K = 256, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 25
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 24
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 40
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 256;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x256_tn) {
  /*
  m = 128, n = 256, overall_K = 256, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 25
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 24
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 40
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 256;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_splitK_h884gemm_64x64x32_splits10, volta884_h884gemm_128x256x256_tt) {
  /*
  m = 128, n = 256, overall_K = 256, splits_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 25
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 24
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 40
  for volta884 it is safe to make sure leading dim are multiple of 8
  */
  const int splits_count = 10;
  const int m = 128;
  const int n = 256;
  const int k = 256;

  /*gemm traits*/
  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;
  /*batched reduction traits*/
  typedef cutlass::reduction::BatchedReductionTraits<half,
    half,
    half,
    half,
    half, /*accumulation type*/
    splits_count,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits;

  run_splitK_gemm<GemmTraits, BatchedReductionTraits>(m, n, k, 8/*partitionK_multiple*/, 1.0f, 0.0f);
}

#endif
