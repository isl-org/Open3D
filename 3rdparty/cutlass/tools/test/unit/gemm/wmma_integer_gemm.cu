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
#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API

#include "cutlass_unit_test.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"

#include "tools/test/unit/gemm/gemm_testbed.h"
#include "tools/test/unit/gemm/integer_gemm.h"

/*
    TEST(TestGroup, TestName)

      - TestGroup should follow this template:
          WmmaIntegerGemm_<CTAShape>_<InstructionShape>_<datatype>_<layout>

      - TestName should follow this template
          wmma_integer_gemm_<ProblemShape>_{optional additional specifier(s)}

      - Shapes should be specified as MxNxK (opposite to the Shape<> definition which is KxNxM)
*/

#ifdef CUTLASS_USE_SUBBYTE_WMMA
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    S4 Integer GEMM Unit Tests
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt4Gemm_32x32x64_8x8x32_s4, wmma_integer_gemm_32x32x64) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 32, 32>,
                                        cutlass::Vector<cutlass::int4_t, 8>,
                                        cutlass::Vector<cutlass::int4_t, 8>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<64, 32, 32>,
                                        cutlass::Shape<32, 8, 8>,
                                        8,
                                        8>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 64);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt4Gemm_32x32x256_8x8x32_s4, wmma_integer_gemm_128x128x256) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<256, 128, 128>,
                                        cutlass::Vector<cutlass::int4_t, 8>,
                                        cutlass::Vector<cutlass::int4_t, 8>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<256, 32, 32>,
                                        cutlass::Shape<32, 8, 8>,
                                        32,
                                        32>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(128, 128, 256);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    U4 Integer GEMM Unit Tests
//
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt4Gemm_32x32x64_8x8x32_u4, wmma_integer_gemm_32x32x64) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<64, 32, 32>,
                                        cutlass::Vector<cutlass::uint4_t, 8>,
                                        cutlass::Vector<cutlass::uint4_t, 8>,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<64, 32, 32>,
                                        cutlass::Shape<32, 8, 8>,
                                        8,
                                        8>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 64);
}
#endif //ifdef CUTLASS_USE_SUBBYTE_WMMA
#ifdef CUTLASS_USE_INT_WMMA

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    S8 Integer GEMM Unit Tests
//
////////////////////////////////////////////////////////////////////////////////////////////////////

//
//  16x16x16
//

TEST(WmmaInt8Gemm_32x32x32_16x16x16_s8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_s8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_s8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_s8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// 32x8x16
//

TEST(WmmaInt8Gemm_32x32x32_32x8x16_s8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_s8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_s8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_s8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// 8x32x16
//

TEST(WmmaInt8Gemm_32x32x32_8x32x16_s8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_s8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_s8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_s8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        signed char,
                                        signed char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    U8 Integer GEMM Unit Tests
//
////////////////////////////////////////////////////////////////////////////////////////////////////

//
//  16x16x16
//

TEST(WmmaInt8Gemm_32x32x32_16x16x16_u8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_u8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_u8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_16x16x16_u8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 16, 16>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// 32x8x16
//

TEST(WmmaInt8Gemm_32x32x32_32x8x16_u8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_u8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_u8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_32x8x16_u8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 8, 32>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// 8x32x16
//

TEST(WmmaInt8Gemm_32x32x32_8x32x16_u8_tn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_u8_tt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_u8_nt, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kRowMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(WmmaInt8Gemm_32x32x32_8x32x16_u8_nn, wmma_integer_gemm_32x32x32) {

  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::MatrixLayout::kColumnMajor,
                                        cutlass::Shape<32, 32, 32>,
                                        unsigned char,
                                        unsigned char,
                                        int,
                                        cutlass::gemm::LinearScaling<int>,
                                        int,
                                        cutlass::Shape<32, 32, 32>,
                                        cutlass::Shape<16, 32, 8>,
                                        4,
                                        4>
      WmmaGemmTraits;
  run_integer_gemm<WmmaGemmTraits>(32, 32, 32);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // ifdef CUTLASS_USE_INT_WMMA
#endif // ifdef CUTLASS_USE_WMMA_API
