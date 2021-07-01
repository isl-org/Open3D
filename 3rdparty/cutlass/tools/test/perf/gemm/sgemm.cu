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

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/reduction/batched_reduction_traits.h"
#include "cutlass/gemm/device_gemm_traits.h"
#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#include "tools/test/perf/gemm/cutlass_dispatch_splitK_PI.h"
#pragma warning( disable : 4503)

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////


/// Profile simple gemm kernels
template <typename OutputTile>
int profile_simple_sgemm_kernel(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config,
  std::string const &name,
  std::string const &algo) {

  typedef perf::GemmProfiler<float, float, float, float, float> SGemmProfiler;

  int results = 0;

  {
    typedef cutlass::gemm::SgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_nt", options, config, algo);
  }

  {
    typedef cutlass::gemm::SgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_nn", options, config, algo);
  }

  {
    typedef cutlass::gemm::SgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_tn", options, config, algo);
  }

  {
    typedef cutlass::gemm::SgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_tt", options, config, algo);
  }

  return results;
}



/// Profile swizzle-raster gemm kernels
template <typename OutputTile>
int profile_swizzle_sgemm_kernel(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config,
  std::string const &name,
  std::string const &algo) {

  typedef perf::GemmProfiler<float, float, float, float, float> SGemmProfiler;

  int results = 0;

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_one_nt", options, config, algo + "_row_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_B_nt", options, config, algo + "_row_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_one_nt", options, config, algo + "_row_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_B_nt", options, config, algo + "_row_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_one_nt", options, config, algo + "_col_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_B_nt", options, config, algo + "_col_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_one_nt", options, config, algo + "_col_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_B_nt", options, config, algo + "_col_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_one_nn", options, config, algo + "_row_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_B_nn", options, config, algo + "_row_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_one_nn", options, config, algo + "_row_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_B_nn", options, config, algo + "_row_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_one_nn", options, config, algo + "_col_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_B_nn", options, config, algo + "_col_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_one_nn", options, config, algo + "_col_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kColumnMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_B_nn", options, config, algo + "_col_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_one_tt", options, config, algo + "_row_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_B_tt", options, config, algo + "_row_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_one_tt", options, config, algo + "_row_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_B_tt", options, config, algo + "_row_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_one_tt", options, config, algo + "_col_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_B_tt", options, config, algo + "_col_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_one_tt", options, config, algo + "_col_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
        cutlass::Shape<8, 8, 8>/*ThreadGemmShape*/,
        1/*kScalarsPerLdgA*/,
        1/*kScalarsPerLdgB*/>
      thisGemmConfig;
    typedef cutlass::gemm::GemmTileTraitsHelperA<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
      GemmTileTraitsHelperA;
    typedef cutlass::gemm::GemmTileTraitsHelperB<cutlass::MatrixLayout::kRowMajor, thisGemmConfig>
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_B_tt", options, config, algo + "_col_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_one_tn", options, config, algo + "_row_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_1_B_tn", options, config, algo + "_row_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_one_tn", options, config, algo + "_row_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_row_2_B_tn", options, config, algo + "_row_2_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_one_tn", options, config, algo + "_col_1_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_1_B_tn", options, config, algo + "_col_1_B");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_one_tn", options, config, algo + "_col_2_one");
  }

  {
    typedef int index;
    typedef cutlass::gemm::SgemmConfig<OutputTile,
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

    typedef typename CutlassDispatchBasic<SgemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_col_2_B_tn", options, config, algo + "_col_2_B");
  }

  return results;
}

/// Profiles all SGEMM tile sizes
int profile_sgemm(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  int results = 0;

  results |= profile_simple_sgemm_kernel<cutlass::Shape<8, 128, 128> >(output, options, config, "sgemm", "128x128");

#ifdef EXHAUSTIVE_PROF
  results |= profile_swizzle_sgemm_kernel<cutlass::Shape<8, 128, 128> >(output, options, config, "sgemm", "128x128");
#endif // defined EXHAUSTIVE_PROF

  return results;
}

struct SgemmRegistrar {
  SgemmRegistrar() { RegisterGemmProfileFunc(profile_sgemm); }
};

volatile SgemmRegistrar _SgemmRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf
