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

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/igemm_traits.h"
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

template <typename OutputTile, int splits_count>
int profile_igemm_splitkpi_kernel(
  TestbenchOutput<GemmProblem> &output,
  TestbenchOptions const &options,
  Config const &config,
  std::string const &name,
  std::string const &algo) {

  typedef perf::GemmProfiler<int8_t, int8_t, int, int, int> GemmProfiler;

  int results = 0;

  {
    /*batched igemm traits*/
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile
    > IgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<int,
      int,
      int,
      int,
      int, /*accumulation type*/
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      cutlass::Shape<1, 1, 2> >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<IgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_nn", options, config, algo + "_splitk_pi");
  }

  {
    /*batched igemm traits*/
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile
    > IgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<int,
      int,
      int,
      int,
      int, /*accumulation type*/
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      cutlass::Shape<1, 1, 2> >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<IgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_nt", options, config, algo + "_splitk_pi");
  }

  {
    /*batched igemm traits*/
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor,
      OutputTile
    > IgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<int,
      int,
      int,
      int,
      int, /*accumulation type*/
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      cutlass::Shape<1, 1, 2> >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<IgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_tn", options, config, algo + "_splitk_pi");
  }

  {
    /*batched igemm traits*/
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile
    > IgemmTraits;
    /*batched reduction traits*/
    typedef cutlass::reduction::BatchedReductionTraits<int,
      int,
      int,
      int,
      int, /*accumulation type*/
      splits_count,
      cutlass::Shape<1, 1, 128>,
      cutlass::Shape<1, 1, 64>,
      cutlass::Shape<1, 1, 2> >
      BatchedReductionTraits;

    // create a device gemm 
    typedef typename cutlass::gemm::SplitkPIGemmTraits<IgemmTraits, BatchedReductionTraits> deviceGemmTraits;
    typedef typename CutlassDispatchSplitKPIGemmBasic<deviceGemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_tt", options, config, algo + "_splitk_pi");
  }

  return results;
}

/// Profiles all SGEMM tile sizes
int profile_igemm_splitkpi(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  int results = 0;
  /*128x128x32*/
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, 8 >(output, options, config, "igemm_128x128x32_splitk_pi_split8", "128x128");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, 16 >(output, options, config, "igemm_128x128x32_splitk_pi_split16", "128x128");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, 32 >(output, options, config, "igemm_128x128x32_splitk_pi_split32", "128x128");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 128, 128>, 64 >(output, options, config, "igemm_128x128x32_splitk_pi_split64", "128x128");

  /*128x64x32*/
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, 8 >(output, options, config, "igemm_128x64x32_splitk_pi_split8", "128x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, 16 >(output, options, config, "igemm_128x64x32_splitk_pi_split16", "128x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, 20 >(output, options, config, "igemm_128x64x32_splitk_pi_split20", "128x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, 32 >(output, options, config, "igemm_128x64x32_splitk_pi_split32", "128x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 128>, 64 >(output, options, config, "igemm_128x64x32_splitk_pi_split64", "128x64");

  /*128x32x32*/
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 32, 128>, 8 >(output, options, config, "igemm_128x32x32_splitk_pi_split8", "128x32");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 32, 128>, 16 >(output, options, config, "igemm_128x32x32_splitk_pi_split16", "128x32");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 32, 128>, 20 >(output, options, config, "igemm_128x32x32_splitk_pi_split20", "128x32");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 32, 128>, 32 >(output, options, config, "igemm_128x32x32_splitk_pi_split32", "128x32");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 32, 128>, 64 >(output, options, config, "igemm_128x32x32_splitk_pi_split64", "128x32");

  /*64x64x32*/
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 64>, 8 >(output, options, config, "igemm_64x64x32_splitk_pi_split8", "64x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 64>, 16 >(output, options, config, "igemm_64x64x32_splitk_pi_split16", "64x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 64>, 20 >(output, options, config, "igemm_64x64x32_splitk_pi_split20", "64x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 64>, 32 >(output, options, config, "igemm_64x64x32_splitk_pi_split32", "64x64");
  results |= profile_igemm_splitkpi_kernel<cutlass::Shape<32, 64, 64>, 64 >(output, options, config, "igemm_64x64x32_splitk_pi_split64", "64x64");

  return results;
}

struct IgemmSplitKPIRegistrar {
  IgemmSplitKPIRegistrar() { RegisterGemmProfileFunc(profile_igemm_splitkpi); }
};

volatile IgemmSplitKPIRegistrar _IgemmSplitKPIRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

#endif // if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))
