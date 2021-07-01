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
#include "cutlass/shape.h"
#include "tools/util/host_tensor.h"
#include "cutlass/reduction/batched_reduction.h"
#include "cutlass/reduction/batched_reduction_traits.h"
#include "tools/test/unit/reduction/test_batched_reduction.h"
#include "tools/test/unit/reduction/batched_reduction_testbed.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Float_batched_reduction_half_alphabeta_float, batched_reduction_128x256x16) {
  /*
  The output matrix is 128x256
  The input matrix is 128x256x16
  The reduction will be applied at the third dim of input matrix
  A is float, Accumulation is float
  alpha and beta are float
  C and D are half
  */


  const int m = 128;
  const int n = 256;
  const int lda = 128;
  const int ldc = 128;
  const int ldd = 128;
  const int reduction_size = 16;
  typedef cutlass::reduction::BatchedReductionTraits<float, /*A*/
    half, /*C*/
    half, /*D*/
    float, /*alpha and beta*/
    float, /*accumulation type*/
    reduction_size,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
  BatchedReductionTraits_16;

  test_batched_reduction<BatchedReductionTraits_16>(m, n, lda, ldc, ldd);

}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Float_batched_reduction_half_alphabeta_half, batched_reduction_128x256x16) {
  /*
  The output matrix is 128x256
  The input matrix is 128x256x16
  The reduction will be applied at the third dim of input matrix
  A is float, Accumulation is float
  alpha and beta are float
  C and D are half
  */


  const int m = 128;
  const int n = 256;
  const int lda = 128;
  const int ldc = 128;
  const int ldd = 128;
  const int reduction_size = 16;
  typedef cutlass::reduction::BatchedReductionTraits<float, /*A*/
    half, /*C*/
    half, /*D*/
    half, /*alpha and beta*/
    float, /*accumulation type*/
    reduction_size,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits_16;

  test_batched_reduction<BatchedReductionTraits_16>(m, n, lda, ldc, ldd);

}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Float_batched_reduction_half_alphabeta_float, batched_reduction_128x64x80) {
  /*
  The output matrix is 128x64
  The input matrix is 128x64x80
  The reduction will be applied at the third dim of input matrix
  */


  const int m = 128;
  const int n = 64;
  const int lda = 128;
  const int ldc = 128;
  const int ldd = 128;
  const int reduction_size = 80;
  typedef cutlass::reduction::BatchedReductionTraits<float, /*A*/
    half, /*C*/
    half, /*D*/
    float, /*alpha and beta*/
    float, /*accumulation type*/
    reduction_size,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits_80;

  test_batched_reduction<BatchedReductionTraits_80>(m, n, lda, ldc, ldd);

}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Float_batched_reduction_half_alphabeta_half, batched_reduction_128x64x80) {
  /*
  The output matrix is 128x64
  The input matrix is 128x64x80
  The reduction will be applied at the third dim of input matrix
  */


  const int m = 128;
  const int n = 64;
  const int lda = 128;
  const int ldc = 128;
  const int ldd = 128;
  const int reduction_size = 80;
  typedef cutlass::reduction::BatchedReductionTraits<float, /*A*/
    half, /*C*/
    half, /*D*/
    half, /*alpha and beta*/
    float, /*accumulation type*/
    reduction_size,
    cutlass::Shape<1, 1, 128>,
    cutlass::Shape<1, 1, 64>,
    cutlass::Shape<1, 1, 2> >
    BatchedReductionTraits_80;

  test_batched_reduction<BatchedReductionTraits_80>(m, n, lda, ldc, ldd);

}
