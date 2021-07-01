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

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))

#include "cutlass_unit_test.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/igemm_traits.h"
#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/run_gemm.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x4_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x32_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x36_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 36);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x64_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x256_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x128x64_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x256x64_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x256x64_nt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x4_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x32_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x36_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 36);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x64_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x256_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x128x64_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x256x64_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x256x64_nn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// NB: I have removed tests in which k=1. These result in the test environment definining matrices
//     in which ld{a,b} = 1 which cannot be launched by cuBLAS.
//
// This problem size remains untested. --akerr
//

TEST(Igemm_128x128x32_float, igemm_128x128x4_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x32_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x36_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 36);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x64_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x256_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x128x64_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x256x64_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x256x64_tn) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x4_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kColumnMajor,
                                     cutlass::Shape<32, 128, 128> , float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x32_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x36_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 36);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x64_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x128x256_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 128, 256);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x128x64_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(256, 128, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_128x256x64_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;
  run_gemm<IgemmTraits>(128, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Igemm_128x128x32_float, igemm_256x256x64_tt) {
  typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
                                     cutlass::MatrixLayout::kRowMajor,
                                     cutlass::Shape<32, 128, 128>, float>
      IgemmTraits;

  run_gemm<IgemmTraits>(256, 256, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))
