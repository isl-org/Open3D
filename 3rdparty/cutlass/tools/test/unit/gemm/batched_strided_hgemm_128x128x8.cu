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


////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_256x384x64x3_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
      HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(256/*m*/, 384/*n*/, 64/*k*/, 3 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_128x384x192x2_nn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(128/*m*/, 384/*n*/, 192/*k*/, 2 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_256x384x64x3_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(256/*m*/, 384/*n*/, 64/*k*/, 3 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_128x384x192x2_nt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(128/*m*/, 384/*n*/, 192/*k*/, 2 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_256x384x64x3_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(256/*m*/, 384/*n*/, 64/*k*/, 3 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_128x384x192x2_tn) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(128/*m*/, 384/*n*/, 192/*k*/, 2 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_256x384x64x3_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(256/*m*/, 384/*n*/, 64/*k*/, 3 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Hgemm_strided_batched_128x128x8, hgemm_128x384x192x2_tt) {
  typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
    HgemmTraits;
  //think about using run_gemm directly
  run_batched_strided_gemm<HgemmTraits>(128/*m*/, 384/*n*/, 192/*k*/, 2 /*batch_size*/);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
