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
#include "tools/test/perf/cutlass_perf_test.h"
#include "tools/test/perf/gemm/gemm_perf_testbed.h"
#include "tools/test/perf/gemm/gemm_profiler.h"
#include "tools/test/perf/gemm/cutlass_dispatch.h"

#pragma warning( disable : 4503)

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DummyT>
int profile_igemm(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {

  typedef perf::GemmProfiler<int8_t, int8_t, int, int, int> GemmProfiler;

  // compute capability check
  if (!options.compute_capability(6, 1)) {
    return 0;
  }

  int results = 0;

  {
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_nt", options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kColumnMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_nn", options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kColumnMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_tn", options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<
      cutlass::MatrixLayout::kRowMajor,
      cutlass::MatrixLayout::kRowMajor
    > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_tt", options, config);
  }

  return results;
}

template <typename DummyT>
int profile_igemm_32x32x128(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {

  typedef perf::GemmProfiler<int8_t, int8_t, int, int, int> GemmProfiler;

  // compute capability check
  if (!options.compute_capability(6, 1)) {
    return 0;
  }

  int results = 0;

  {
    typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<128, 32, 32>, int,
            cutlass::gemm::LinearScaling<int>, cutlass::Shape<32, 8, 4> > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_32x32x128_nn",
            options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
            cutlass::MatrixLayout::kRowMajor, cutlass::Shape<128, 32, 32>, int,
            cutlass::gemm::LinearScaling<int>, cutlass::Shape<32, 8, 4> > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_32x32x128_nt",
            options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kColumnMajor, cutlass::Shape<128, 32, 32>, int,
            cutlass::gemm::LinearScaling<int>, cutlass::Shape<32, 8, 4> > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, "igemm_32x32x128_tn",
            options, config);
  }

  {
    typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kRowMajor,
            cutlass::MatrixLayout::kRowMajor, cutlass::Shape<128, 32, 32>, int,
            cutlass::gemm::LinearScaling<int>, cutlass::Shape<32, 8, 4> > GemmTraits;

    typedef typename CutlassDispatchBasic<GemmTraits>::Dispatch Dispatch;

    results = profile_gemm<Dispatch, GemmProfiler>(output, "igemm_32x32x128_tt",
            options, config);
  }

  return results;
}



struct IgemmRegistrar {
  IgemmRegistrar()
  {
    RegisterGemmProfileFunc(profile_igemm<void>);

#ifdef EXHAUSTIVE_PROF
    RegisterGemmProfileFunc(profile_igemm_32x32x128<void>);
#endif // defined EXHAUSTIVE_PROF

  }
};

volatile IgemmRegistrar _IgemmRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

#endif // if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))
