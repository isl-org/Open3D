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
#include "cutlass/gemm/dgemm_traits.h"

#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"
#pragma warning( disable : 4503)
namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

int profile_dgemm(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  typedef perf::GemmProfiler<double, double, double, double, double> GemmProfiler;

  int results = 0;

  // compute capability check
  if (!options.compute_capability(6, 0)) {
    return 0;
  }

  {
    typedef cutlass::gemm::DgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "dgemm_nt", options, config);
  }

  {
    typedef cutlass::gemm::DgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "dgemm_nn", options, config);
  }

  {
    typedef cutlass::gemm::DgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "dgemm_tn", options, config);
  }

  {
    typedef cutlass::gemm::DgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "dgemm_tt", options, config);
  }

  return results;
}

struct DgemmRegistrar {
  DgemmRegistrar() { RegisterGemmProfileFunc(profile_dgemm); }
};

volatile DgemmRegistrar _DgemmRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf
