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

template <typename OutputTile, typename threadGemmShape, typename threadReductionShape, int splits_count>
int profile_sgemm_splitkpi_kernel(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config,
  std::string const &name,
  std::string const &algo) {

  typedef perf::GemmProfiler<float, float, float, float, float> SGemmProfiler;

  int results = 0;

  {
    /*batched sgemm traits*/
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor, OutputTile,
      cutlass::gemm::LinearScaling<float>, threadGemmShape>
      SgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<float,
      float,
      float,
      float,
      float,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<SgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_nn", options, config, algo + "_splitk_pi");
  }

  {
    /*batched sgemm traits*/
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor, OutputTile,
      cutlass::gemm::LinearScaling<float>, threadGemmShape>
      SgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<float,
      float,
      float,
      float,
      float,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<SgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_nt", options, config, algo + "_splitk_pi");
  }

  {
    /*batched sgemm traits*/
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor, OutputTile,
      cutlass::gemm::LinearScaling<float>, threadGemmShape>
      SgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<float,
      float,
      float,
      float,
      float,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<SgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_tn", options, config, algo + "_splitk_pi");
  }

  {
    /*batched sgemm traits*/
    typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor, OutputTile,
      cutlass::gemm::LinearScaling<float>, threadGemmShape>
      SgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<float,
      float,
      float,
      float,
      float,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<SgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, SGemmProfiler>(output, name + "_tt", options, config, algo + "_splitk_pi");
  }


  return results;
}

/// Profiles all SGEMM tile sizes
int profile_sgemm_splitkpi(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  int results = 0;
  /*128x128x8*/
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "sgemm_128x128x8_splitk_pi_split5", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "sgemm_128x128x8_splitk_pi_split8", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_128x128x8_splitk_pi_split10", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "sgemm_128x128x8_splitk_pi_split16", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "sgemm_128x128x8_splitk_pi_split20", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "sgemm_128x128x8_splitk_pi_split24", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "sgemm_128x128x8_splitk_pi_split28", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "sgemm_128x128x8_splitk_pi_split32", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "sgemm_128x128x8_splitk_pi_split40", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "sgemm_128x128x8_splitk_pi_split48", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "sgemm_128x128x8_splitk_pi_split56", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "sgemm_128x128x8_splitk_pi_split64", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "sgemm_128x128x8_splitk_pi_split72", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "sgemm_128x128x8_splitk_pi_split80", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "sgemm_128x128x8_splitk_pi_split88", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "sgemm_128x128x8_splitk_pi_split96", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "sgemm_128x128x8_splitk_pi_split104", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "sgemm_128x128x8_splitk_pi_split112", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "sgemm_128x128x8_splitk_pi_split120", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "sgemm_128x128x8_splitk_pi_split128", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "sgemm_128x128x8_splitk_pi_split136", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "sgemm_128x128x8_splitk_pi_split144", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "sgemm_128x128x8_splitk_pi_split152", "128x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "sgemm_128x128x8_splitk_pi_split160", "128x128");

#ifdef EXHAUSTIVE_PROF
  /*128x64x8*/
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "sgemm_128x64x8_splitk_pi_split5", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "sgemm_128x64x8_splitk_pi_split8", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_128x64x8_splitk_pi_split10", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "sgemm_128x64x8_splitk_pi_split16", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "sgemm_128x64x8_splitk_pi_split20", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "sgemm_128x64x8_splitk_pi_split24", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "sgemm_128x64x8_splitk_pi_split28", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "sgemm_128x64x8_splitk_pi_split32", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "sgemm_128x64x8_splitk_pi_split40", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "sgemm_128x64x8_splitk_pi_split48", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "sgemm_128x64x8_splitk_pi_split56", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "sgemm_128x64x8_splitk_pi_split64", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "sgemm_128x64x8_splitk_pi_split72", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "sgemm_128x64x8_splitk_pi_split80", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "sgemm_128x64x8_splitk_pi_split88", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "sgemm_128x64x8_splitk_pi_split96", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "sgemm_128x64x8_splitk_pi_split104", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "sgemm_128x64x8_splitk_pi_split112", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "sgemm_128x64x8_splitk_pi_split120", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "sgemm_128x64x8_splitk_pi_split128", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "sgemm_128x64x8_splitk_pi_split136", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "sgemm_128x64x8_splitk_pi_split144", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "sgemm_128x64x8_splitk_pi_split152", "128x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "sgemm_128x64x8_splitk_pi_split160", "128x64");

  /*128x32x8*/ 
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 5  >(output, options, config, "sgemm_128x32x8_splitk_pi_split5", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 8  >(output, options, config, "sgemm_128x32x8_splitk_pi_split8", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_128x32x8_splitk_pi_split10", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "sgemm_128x32x8_splitk_pi_split16", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "sgemm_128x32x8_splitk_pi_split20", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "sgemm_128x32x8_splitk_pi_split24", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "sgemm_128x32x8_splitk_pi_split28", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "sgemm_128x32x8_splitk_pi_split32", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "sgemm_128x32x8_splitk_pi_split40", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "sgemm_128x32x8_splitk_pi_split48", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "sgemm_128x32x8_splitk_pi_split56", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "sgemm_128x32x8_splitk_pi_split64", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "sgemm_128x32x8_splitk_pi_split72", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "sgemm_128x32x8_splitk_pi_split80", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "sgemm_128x32x8_splitk_pi_split88", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "sgemm_128x32x8_splitk_pi_split96", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "sgemm_128x32x8_splitk_pi_split104", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "sgemm_128x32x8_splitk_pi_split112", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "sgemm_128x32x8_splitk_pi_split120", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "sgemm_128x32x8_splitk_pi_split128", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "sgemm_128x32x8_splitk_pi_split136", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "sgemm_128x32x8_splitk_pi_split144", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "sgemm_128x32x8_splitk_pi_split152", "128x32");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 32, 128>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "sgemm_128x32x8_splitk_pi_split160", "128x32");
  
  /*64x128*/
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "sgemm_64x128x8_splitk_pi_split5", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "sgemm_64x128x8_splitk_pi_split8", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_64x128x8_splitk_pi_split10", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "sgemm_64x128x8_splitk_pi_split16", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "sgemm_64x128x8_splitk_pi_split20", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "sgemm_64x128x8_splitk_pi_split24", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "sgemm_64x128x8_splitk_pi_split28", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "sgemm_64x128x8_splitk_pi_split32", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "sgemm_64x128x8_splitk_pi_split40", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "sgemm_64x128x8_splitk_pi_split48", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "sgemm_64x128x8_splitk_pi_split56", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "sgemm_64x128x8_splitk_pi_split64", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "sgemm_64x128x8_splitk_pi_split72", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "sgemm_64x128x8_splitk_pi_split80", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "sgemm_64x128x8_splitk_pi_split88", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "sgemm_64x128x8_splitk_pi_split96", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "sgemm_64x128x8_splitk_pi_split104", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "sgemm_64x128x8_splitk_pi_split112", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "sgemm_64x128x8_splitk_pi_split120", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "sgemm_64x128x8_splitk_pi_split128", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "sgemm_64x128x8_splitk_pi_split136", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "sgemm_64x128x8_splitk_pi_split144", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "sgemm_64x128x8_splitk_pi_split152", "64x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "sgemm_64x128x8_splitk_pi_split160", "64x128");
  
  /*32x128*/
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "sgemm_32x128x8_splitk_pi_split5", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "sgemm_32x128x8_splitk_pi_split8", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_32x128x8_splitk_pi_split10", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "sgemm_32x128x8_splitk_pi_split16", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "sgemm_32x128x8_splitk_pi_split20", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "sgemm_32x128x8_splitk_pi_split24", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "sgemm_32x128x8_splitk_pi_split28", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "sgemm_32x128x8_splitk_pi_split32", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "sgemm_32x128x8_splitk_pi_split40", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "sgemm_32x128x8_splitk_pi_split48", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "sgemm_32x128x8_splitk_pi_split56", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "sgemm_32x128x8_splitk_pi_split64", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "sgemm_32x128x8_splitk_pi_split72", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "sgemm_32x128x8_splitk_pi_split80", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "sgemm_32x128x8_splitk_pi_split88", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "sgemm_32x128x8_splitk_pi_split96", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "sgemm_32x128x8_splitk_pi_split104", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "sgemm_32x128x8_splitk_pi_split112", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "sgemm_32x128x8_splitk_pi_split120", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "sgemm_32x128x8_splitk_pi_split128", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "sgemm_32x128x8_splitk_pi_split136", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "sgemm_32x128x8_splitk_pi_split144", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "sgemm_32x128x8_splitk_pi_split152", "32x128");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 128, 32>, cutlass::Shape<8, 8, 4>, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "sgemm_32x128x8_splitk_pi_split160", "32x128");

  /*64x64*/
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "sgemm_64x64x8_splitk_pi_split5", "64x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "sgemm_64x64x8_splitk_pi_split8", "64x64");
  results |= profile_sgemm_splitkpi_kernel<cutlass::Shape<8, 64, 64>, cutlass::Shape<8, 8, 8>, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "sgemm_64x64x8_splitk_pi_split10", "64x64");

#endif //#ifdef EXHAUSTIVE_PROF

  return results;
}

struct SgemmSplitKPIRegistrar {
  SgemmSplitKPIRegistrar() { RegisterGemmProfileFunc(profile_sgemm_splitkpi); }
};

volatile SgemmSplitKPIRegistrar _SgemmSplitKPIRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf
