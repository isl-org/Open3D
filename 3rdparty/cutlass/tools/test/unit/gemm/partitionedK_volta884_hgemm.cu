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

#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

#if CUTLASS_ENABLE_TENSOR_CORE_MMA

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x88x10_nn) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 88, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 88;
  int partitionK_count = 10;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x88x10_nt) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 88, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 88;
  int partitionK_count = 10;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x88x10_tn) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 88, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 88;
  int partitionK_count = 10;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x88x10_tt) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 88, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 88;
  int partitionK_count = 10;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x128x10_nn) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 128, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 12. 
  But if we require the partition mulitple to be 8, the first 9 partition 
  k = k - (k % partition_mulitiple) = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 56
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 128;
  int partitionK_count = 10;
  int partitionK_multiple = 8;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count, partitionK_multiple);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x128x10_nt) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 128, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 12.
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 56
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 128;
  int partitionK_count = 10;
  int partitionK_multiple = 8;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count, partitionK_multiple);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x128x10_tn) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 128, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 12.
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 56
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 128;
  int partitionK_count = 10;
  int partitionK_multiple = 8;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count, partitionK_multiple);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_partitionedK_64x64x32, volta884_h884gemm_128x256x128x10_tt) {
  /*
  for example
  partitionedK gemm, m = 128, n = 256, overall_K = 128, partitionK_count = 10
  for the first 9 partition k = overall_k / partitionK_count = 12.
  But if we require the partition mulitple to be 8, the first 9 partition
  k = k - (k % partition_mulitiple) = 8
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 56
  for volta884 it is safe to make sure leading dim are multiple of 8
  */

  int m = 128;
  int n = 256;
  int overall_k = 128;
  int partitionK_count = 10;
  int partitionK_multiple = 8;

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

  run_partitioned_k_gemm<GemmTraits>(m, n, overall_k, partitionK_count, partitionK_multiple);
}

#endif // if defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
