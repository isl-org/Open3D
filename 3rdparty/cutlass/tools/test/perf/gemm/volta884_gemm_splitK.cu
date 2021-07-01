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


////////////////////////////////////////////////////////////////////////////////////////////////////

#include "tools/test/perf/gemm/cutlass_volta884_dispatch_splitK_PI.h"
#include "cutlass/reduction/batched_reduction_traits.h"
#include "cutlass/gemm/device_gemm_traits.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename OutputTile, typename AccumHostType, typename threadReductionShape, int splits_count>
int profile_volta884_gemm_splitkpi_kernel(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config,
  std::string const &name,
  std::string const &algo) {

  int results = 0;

  // compute capability check
  if (!options.compute_capability(7, 0)) {
    return 0;
  }

  typedef typename cutlass::TypeTraits<AccumHostType>::device_type AccumDevType;

#if CUTLASS_ENABLE_TENSOR_CORE_MMA
  typedef perf::GemmProfiler<
    cutlass::half_t,
    cutlass::half_t,
    cutlass::half_t,
    AccumHostType,
    AccumHostType> GemmProfiler;

  {
    typedef cutlass::gemm::Volta884GemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile,
      cutlass::Shape<32, 64, 64>,
      AccumDevType,
      AccumDevType,
      AccumDevType,
      2
    > GemmTraits;

    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<AccumDevType,
      half,
      half,
      AccumDevType,
      AccumDevType,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<GemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef Volta884GemmDispatchSplitKPI<deviceGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_nn", options, config, algo);
  }

  {
    typedef cutlass::gemm::Volta884GemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile,
      cutlass::Shape<32, 64, 64>,
      AccumDevType,
      AccumDevType,
      AccumDevType,
      2
    > GemmTraits;

    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<AccumDevType,
      half,
      half,
      AccumDevType,
      AccumDevType,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<GemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef Volta884GemmDispatchSplitKPI<deviceGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_nt", options, config, algo);
  }

  {
    typedef cutlass::gemm::Volta884GemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile,
      cutlass::Shape<32, 64, 64>,
      AccumDevType,
      AccumDevType,
      AccumDevType,
      2
    > GemmTraits;

    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<AccumDevType,
      half,
      half,
      AccumDevType,
      AccumDevType,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<GemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef Volta884GemmDispatchSplitKPI<deviceGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_tn", options, config, algo);
}

  {
    typedef cutlass::gemm::Volta884GemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile,
      cutlass::Shape<32, 64, 64>,
      AccumDevType,
      AccumDevType,
      AccumDevType,
      2
    > GemmTraits;

    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<AccumDevType,
      half,
      half,
      AccumDevType,
      AccumDevType,
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      threadReductionShape >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<GemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef Volta884GemmDispatchSplitKPI<deviceGemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_tt", options, config, algo);
  }

  #endif // if defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int profile_volta884_gemm_splitkpi(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  int results = 0;

  //results |= profile_volta884_gemm_kernel<cutlass::Shape<32, 128, 128>, float >(output, options, config, "s884gemm", "128x128");

  // half accum
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits5", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits8", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits10", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits16", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits20", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits24", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits28", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits32", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits40", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits48", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits56", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits64", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits72", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits80", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits88", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits96", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits104", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits112", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits120", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits128", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits136", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits144", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits152", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, cutlass::half_t, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "h884gemm_128x128x32_splitk_pi_splits160", "128x128");

  // float accum
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits5", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits8", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits10", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits16", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits20", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits24", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits28", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits32", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits40", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits48", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits56", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits64", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits72", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits80", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits88", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits96", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits104", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits112", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits120", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits128", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits136", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits144", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits152", "128x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, float, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "s884gemm_128x128x32_splitk_pi_splits160", "128x128");

#ifdef EXHAUSTIVE_PROF
  // float accum 128x64
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits5", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits8", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits10", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits16", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits20", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits24", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits28", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits32", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits40", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits48", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits56", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits64", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits72", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits80", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits88", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits96", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits104", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits112", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits120", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits128", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits136", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits144", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits152", "128x64");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, float, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "s884gemm_128x64x32_splitk_pi_splits160", "128x64");

  // float accum 64x128
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 5 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits5", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 8 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits8", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 10 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits10", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 16 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits16", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 20 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits20", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 24 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits24", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 28 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits28", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 32 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits32", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 40 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits40", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 48 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits48", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 56 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits56", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 64 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits64", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 72 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits72", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 2>, 80 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits80", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 88 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits88", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 96 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits96", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 104 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits104", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 112 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits112", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 120 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits120", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 128 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits128", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 136 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits136", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 144 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits144", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 152 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits152", "64x128");
  results |= profile_volta884_gemm_splitkpi_kernel<cutlass::Shape<32, 128, 64>, float, cutlass::Shape<1, 1, 1>, 160 >(output, options, config, "s884gemm_64x128x32_splitk_pi_splits160", "64x128");
#endif //#ifdef EXHAUSTIVE_PROF
  return results;
}

struct Volta884GemmSplitKPIRegistrar {
  Volta884GemmSplitKPIRegistrar() { RegisterGemmProfileFunc(profile_volta884_gemm_splitkpi); }
};

volatile Volta884GemmSplitKPIRegistrar _Volta884GemmSplitKPIRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

////////////////////////////////////////////////////////////////////////////////////////////////////
