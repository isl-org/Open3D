/***************************************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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
#include "cutlass/gemm/dgemm_traits.h"
#include "cutlass/gemm/igemm_traits.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/gemm_nvrtc.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Dgemm_nvrtc_64x32x8, dgemm_nvrtc_64x32x8_nt) {

  typedef cutlass::gemm::DgemmTraits<cutlass::MatrixLayout::kColumnMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<8, 32, 64> > GemmTraits;
  static char const *gemm_traits = "cutlass::gemm::DgemmTraits<cutlass::MatrixLayout::kColumnMajor, cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 32, 64> >";
  run_gemm_nvrtc<GemmTraits>(gemm_traits, 64, 32, 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))

TEST(Igemm__nvrtc_128x128x32, igemm_nvrtc_256x256x64_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
    cutlass::MatrixLayout::kRowMajor,
    cutlass::Shape<32, 128, 128>, int, cutlass::gemm::LinearScaling<int> >
    IgemmTraits;
  static char const *gemm_traits = "cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor, cutlass::MatrixLayout::kRowMajor, cutlass::Shape<32, 128, 128>, int, cutlass::gemm::LinearScaling<int> >";
  run_gemm_nvrtc<IgemmTraits>(gemm_traits, 256, 256, 64);
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_nvrtc_128x128x8, sgemm_nvrtc_128x112x16_alpha2_beta1_nt) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
      SgemmTraits;
  static char const *gemm_traits = "cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor, cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >";
  run_gemm_nvrtc<SgemmTraits>(gemm_traits, 128, 112, 16, 2.f, 1.f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
