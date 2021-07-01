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

#include "tools/test/perf/gemm/cutlass_volta884_dispatch.h"

#ifdef EXHAUSTIVE_PROF

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {
template <typename OutputTile, typename AccumHostType>
int profile_volta884_gemm_cta_rasterization_nt_kernel(
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
    AccumHostType,
    AccumHostType,
    AccumHostType> GemmProfiler;

  {
    typedef cutlass::gemm::Volta884GemmTraits<
      cutlass::MatrixLayout::kColumnMajor,
      cutlass::MatrixLayout::kRowMajor,
      OutputTile,
      cutlass::Shape<32, 64, 64>,
      AccumDevType,
      AccumDevType,
      AccumDevType,
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_row_1_one_nt", options, config, algo + "_row_1_one");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::RowMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_row_1_B_nt", options, config, algo + "_row_1_B");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_row_2_one_nt", options, config, algo + "_row_2_one");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::RowMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_row_2_B_nt", options, config, algo + "_row_2_B");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::OneDirection>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_col_1_one_nt", options, config, algo + "_col_1_one");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::ColumnMajorBlockSwizzle<1, cutlass::gemm::swizzleDirection::Boustrophedon>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_col_1_B_nt", options, config, algo + "_col_1_B");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::OneDirection>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_col_2_one_nt", options, config, algo + "_col_2_one");
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
      2,
      cutlass::gemm::LinearScaling<AccumDevType>,
      typename cutlass::gemm::ColumnMajorBlockSwizzle<2, cutlass::gemm::swizzleDirection::Boustrophedon>
    > GemmTraits;

    typedef Volta884GemmDispatch<GemmTraits> Dispatch;

    results |= profile_gemm<Dispatch, GemmProfiler>(output, name + "_col_2_B_nt", options, config, algo + "_col_2_B");
  }
#endif // if defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)

  return results;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int profile_volta884_gemm_cta_rasterization_nt(TestbenchOutput<GemmProblem> &output, TestbenchOptions const &options, Config const &config) {
  int results = 0;

  results |= profile_volta884_gemm_cta_rasterization_nt_kernel<cutlass::Shape<32, 128, 128>, float >(output, options, config, "s884gemm", "128x128");

  results |= profile_volta884_gemm_cta_rasterization_nt_kernel<cutlass::Shape<32, 128, 256>, float >(output, options, config, "s884gemm_256x128", "256x128");

  results |= profile_volta884_gemm_cta_rasterization_nt_kernel<cutlass::Shape<32, 64, 128>, float >(output, options, config, "s884gemm_128x64", "128x64");

  results |= profile_volta884_gemm_cta_rasterization_nt_kernel<cutlass::Shape<32, 64, 64>, float >(output, options, config, "s884gemm_64x64", "64x64");

  return results;
}

struct Volta884GemmCTARasterizationNTRegistrar {
  Volta884GemmCTARasterizationNTRegistrar() { RegisterGemmProfileFunc(profile_volta884_gemm_cta_rasterization_nt); }
};

volatile Volta884GemmCTARasterizationNTRegistrar _Volta884CTARasterizationNTGemmRegistrar;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace perf

#endif // if defined(EXHAUSTIVE_PROF)

////////////////////////////////////////////////////////////////////////////////////////////////////
