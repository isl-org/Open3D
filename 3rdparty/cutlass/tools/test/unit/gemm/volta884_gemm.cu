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
//
// Very small warp sizes
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_32x32x32_nn, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 32, 32>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_64x64x32_32x32x32_nt, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 32, 32>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_64x64x32_32x32x32_tn, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 32, 32>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_64x64x32_32x32x32_tt, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 32, 32>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Short compile time - s884gemm
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_128x128x32_nt, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_128x128x32_tn, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_128x128x32_tt, short_480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Contiguous - s884gemm
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nt, 64x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nt, 64x64x30_residue) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 30);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nt, 64x64x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nt, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nt, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_nt, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_nt, 128x64x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 128x128x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 384x256x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(384, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 392x264x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 392x264x192) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 192);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nt, 480x280x223) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 223);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Crosswise
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tn, 64x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tn, 64x64x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tn, 64x64x24_residue) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tn, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tn, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_tn, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_tn, 128x64x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 128x128x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 384x256x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(384, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 392x264x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 392x264x192) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 192);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Congruous-Crosswise
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nn, 64x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nn, 64x64x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nn, 64x64x24_residue) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nn, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_nn, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_nn, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_nn, 128x64x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 128x128x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 384x256x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(384, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 392x264x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 392x264x192) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 192);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_nn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Crosswise-Congruous
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tt, 64x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tt, 64x64x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tt, 64x64x24_residue) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(64, 64, 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tt, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_64x64x32_tt, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_tt, 128x64x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x64x32_tt, 128x64x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 64, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 128x128x32) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 128x128x128) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(128, 128, 128);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 384x256x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(384, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 392x264x64) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 392x264x192) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(392, 264, 192);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_128x128x32_tt, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    float,
    float,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FP32 accumulation, FP16 output
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Volta884_f16_s884gemm_f16_128x128x32_nn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_128x128x32_nt, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_128x128x32_tn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_128x128x32_tt, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_64x64x32_nt, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 64, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_64x128x32_nt, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 64>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_128x64x32_tn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 64, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_256x128x32_tn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 128, 256>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}

TEST(Volta884_f16_s884gemm_f16_128x256x32_tn, 480x280x224) {

  typedef cutlass::gemm::Volta884GemmTraits<
    cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kColumnMajor,
    cutlass::Shape<32, 256, 128>,
    cutlass::Shape<32, 64, 64>,
    float,
    half,
    half,
    2
  > GemmTraits;

  run_gemm<GemmTraits>(480, 280, 224);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // if defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
