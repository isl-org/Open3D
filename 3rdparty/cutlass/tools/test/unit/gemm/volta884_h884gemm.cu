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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Contiguous - h884gemm
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_h884gemm_64x64x32_nt, 520x264x136) {

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

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x64x32_nt, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_64x128x32_nt, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x128x32_nt, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_256x128x32_nt, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 256>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x256x32_nt, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 256, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_64x64x32_tn, 520x264x136) {

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

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x64x32_tn, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_64x128x32_tn, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x128x32_tn, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_256x128x32_tn, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 256>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

TEST(Volta884_h884gemm_128x256x32_tn, 520x264x136) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 256, 128>,
    cutlass::Shape<32, 64, 64>,
    half,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(520, 264, 136);
}

#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

#endif // defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)

