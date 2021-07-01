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

#pragma warning( disable : 4503)

////////////////////////////////////////////////////////////////////////////////////////////////////
//Row Major Swizzle
TEST(Sgemm_512x256x16_swizzle, sgemm_128x128x16_tn_RowMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
      cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
      1/*kScalarsPerLdgA*/,
      1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
      SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_512x256x16_swizzle, sgemm_128x64x16_tn_RowMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x64x16_tn_RowMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol2) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol2) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol3) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol3) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//Row Major Swizzle Boustrophedon
TEST(Sgemm_512x256x16_swizzle, sgemm_128x128x16_tn_RowMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_512x256x16_swizzle, sgemm_128x64x16_tn_RowMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x64x16_tn_RowMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol2_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol2_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol3_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_RowMajorSwizzle_groupCol3_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::RowMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//Column Major Swizzle

TEST(Sgemm_512x256x16_swizzle, sgemm_128x128x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x128x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_512x256x16_swizzle, sgemm_128x64x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x64x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol2) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol2) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol3) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol3) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::OneDirection>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//Column Major Swizzle

TEST(Sgemm_512x256x16_swizzle, sgemm_128x128x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x128x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 128, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_512x256x16_swizzle, sgemm_128x64x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(512, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_516x260x16_swizzle, sgemm_128x64x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 64, 128>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(516, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol2_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol2_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1030, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1024x256x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol3_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 256, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Sgemm_1030x260x16_swizzle, sgemm_64x32x16_tn_ColumnMajorSwizzle_groupCol3_Boustrophedon) {
  typedef int index;
  typedef cutlass::gemm::SgemmConfig<cutlass::Shape<16, 32, 64>/*OutputTile*/,
    cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
    1/*kScalarsPerLdgA*/,
    1/*kScalarsPerLdgB*/>
    thisGemmConfig;
  typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
    GemmTileTraitsHelperA;
  typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
    GemmTileTraitsHelperB;
  typedef cutlass::gemm::SimplifiedGemmTraitsHelper<GemmTileTraitsHelperA, GemmTileTraitsHelperB, index>
    Helper;
  typedef cutlass::gemm::LinearScaling<float>
    EpilogueFunctor;
  typedef cutlass::gemm::SimplifiedGemmEpilogueTraits<thisGemmConfig, EpilogueFunctor, index>
    GemmEpilogueTraits;
  typedef cutlass::gemm::ClearAccumulators<typename thisGemmConfig::Accumulators::Element>
    ClearAccumulators;

  typedef cutlass::gemm::GemmTraits<
    thisGemmConfig,
    typename Helper::GlobalLoadStreamA,
    typename Helper::GlobalLoadStreamB,
    typename Helper::SharedLoadStreamA,
    typename Helper::SharedLoadStreamB,
    typename cutlass::gemm::GemmEpilogue<GemmEpilogueTraits>,
    typename cutlass::gemm::ColumnMajorBlockSwizzle<3, cutlass::gemm::swizzleDirection::Boustrophedon>,
    index,
    ClearAccumulators
  >
    SgemmTraits;
  //
  run_gemm<SgemmTraits>(1024, 260, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


