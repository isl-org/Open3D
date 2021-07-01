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
TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x8_nn) {
  /*
  for example
  partitionedK sgemm, m = 128, n = 256, overall_K = 100, partitionK_count = 8
  for the first 7 partition k = overall_k / partitionK_count = 12
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  */

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x175x8_nn) {

  int m = 128;
  int n = 256;
  int overall_k = 175;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x20x3_nn) {

  int m = 10;
  int n = 12;
  int overall_k = 20;
  int partitionK_count = 3;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x60x8_nn) {

  int m = 10;
  int n = 12;
  int overall_k = 60;
  int partitionK_count = 8;


  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x4_nn) {

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 4;


  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x8_nt) {
  /*
  for example
  partitionedK sgemm, m = 128, n = 256, overall_K = 100, partitionK_count = 8
  for the first 7 partition k = overall_k / partitionK_count = 12
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  */

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x175x8_nt) {

  int m = 128;
  int n = 256;
  int overall_k = 175;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x20x3_nt) {

  int m = 10;
  int n = 12;
  int overall_k = 20;
  int partitionK_count = 3;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x60x8_nt) {

  int m = 10;
  int n = 12;
  int overall_k = 60;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x4_nt) {

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 4;


  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x8_tn) {
  /*
  for example
  partitionedK sgemm, m = 128, n = 256, overall_K = 100, partitionK_count = 8
  for the first 7 partition k = overall_k / partitionK_count = 12
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  */

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x175x8_tn) {

  int m = 128;
  int n = 256;
  int overall_k = 175;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x20x3_tn) {

  int m = 10;
  int n = 12;
  int overall_k = 20;
  int partitionK_count = 3;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x60x8_tn) {

  int m = 10;
  int n = 12;
  int overall_k = 60;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x4_tn) {

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 4;


  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x8_tt) {
  /*
  for example
  partitionedK sgemm, m = 128, n = 256, overall_K = 100, partitionK_count = 8
  for the first 7 partition k = overall_k / partitionK_count = 12
  for the last partition last_k = overall_k - (partitionK_count - 1) * k = 16
  */

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x175x8_tt) {

  int m = 128;
  int n = 256;
  int overall_k = 175;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x20x3_tt) {

  int m = 10;
  int n = 12;
  int overall_k = 20;
  int partitionK_count = 3;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_10x12x60x8_tt) {

  int m = 10;
  int n = 12;
  int overall_k = 60;
  int partitionK_count = 8;

  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_partitionedK_128x128x8, sgemm_128x256x100x4_tt) {

  int m = 128;
  int n = 256;
  int overall_k = 100;
  int partitionK_count = 4;


  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    SgemmTraits;

  run_partitioned_k_gemm<SgemmTraits>(m, n, overall_k, partitionK_count);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
