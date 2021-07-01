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
#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include "cutlass/util/platform.h"
#if defined(CUTLASS_OS_WINDOWS)
#include <Windows.h>
#else
// needed for sleep
#include <unistd.h>
#endif

#include "tools/test/perf/gemm/gemm_perf_testbed.h"
#include "tools/test/perf/testbench_configs.h"
#include "tools/test/perf/testbench_options.h"
#include "tools/test/perf/testbench_output.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance measuring testbed
template <typename AType,
          typename BType,
          typename CType,
          typename AccumulatorType,
          typename ScalarType>
class GemmProfiler {
 public:
  /// Test environment
  typedef GemmTestbed<AType, BType, CType, AccumulatorType, ScalarType> PerfTestbed;

 private:
  //
  // Data members
  //

  /// Reference to TestbenchOutput instance
  TestbenchOutput<GemmProblem> &output;

  /// Reference to options object
  TestbenchOptions const &options;

  // Reference to config object
  Config const &config;

  /// Performance test environment
  PerfTestbed testbed;

  /// Kernel name
  std::string kernel_name;

  /// Cutlass algorithm
  std::string cutlass_algo;

  /// Timing events
  cudaEvent_t events[2];

 public:
  /// Delays
  static void pause(int seconds) {
#if defined(WIN32)
    Sleep(1000 * seconds);
#else
    sleep(seconds);
#endif
  }

 public:
  //
  // Methods
  //

  /// Constructs performance testebed
  GemmProfiler(TestbenchOutput<GemmProblem> &_output,
               std::string const &_kernel_name,
               std::string const &_cutlass_algo,
               TestbenchOptions const &_options,
               Config const &_config)
      : output(_output),
        options(_options),
        config(_config),
        kernel_name(_kernel_name),
        cutlass_algo(_cutlass_algo),
        testbed(_options.initial_distribution) {
    for (int i = 0; i < 2; ++i) {
      cudaError_t result = cudaEventCreate(&events[i]);
      if (result != cudaSuccess) {
        throw std::runtime_error("GemmPerfTestbed() failed to create CUDA events");
      }
    }
  }

  ~GemmProfiler() {}

  /// Writes the workspace to text files
  void write_problem(Provider::Kind provider, std::string const &kernel_name) {
    std::stringstream base_filename;

    base_filename << provider << "_" << kernel_name << "_" << testbed.M() << "x" << testbed.N()
                  << "x" << testbed.K();

    std::string results_name = base_filename.str() + "_results.txt";
    std::string errors_name = base_filename.str() + "_errors.txt";

    std::ofstream results(results_name.c_str());
    std::ofstream errors(errors_name.c_str());
    testbed.write_problem(results, errors);
  }

  /// Profiles Cutlass
  template <typename CutlassDispatch>
  PerformanceResult<GemmProblem> execute_cutlass(GemmProblem const &problem,
                                                 cublasGemmAlgo_t algorithm) {
    PerformanceResult<GemmProblem> result(
      Provider::Cutlass
      , kernel_name
      , problem
    );
    
    result.disposition = Disposition::NotVerified;
    
    if (options.dry_run) {
      result.disposition = Disposition::NotRun;
      return result;
    }

    if (CutlassDispatch::kRunCuBLAS) {
#if CUTLASS_ENABLE_CUBLAS
      testbed.compute_reference(algorithm);

      if (cudaDeviceSynchronize() != cudaSuccess) {
        result.disposition = Disposition::NotVerified;
        return result;
      }
#endif
    }

    CutlassDispatch *dispatch_ptr;

    // check to see if we need to launch batched strided gemm
    if (testbed.batch_count() == 1) {
      dispatch_ptr = new CutlassDispatch(testbed.M(),
        testbed.N(),
        testbed.K(),
        testbed.alpha(),
        testbed.ptr_A(),
        testbed.lda(),
        testbed.ptr_B(),
        testbed.ldb(),
        testbed.beta(),
        testbed.ptr_C_initial(),
        testbed.ldc(),
        testbed.ptr_experimental(),
        testbed.ldc());

      dispatch_ptr->operator()();
    }
    else {
      dispatch_ptr = new CutlassDispatch(testbed.M(),
        testbed.N(),
        testbed.K(),
        testbed.alpha(),
        testbed.ptr_A(),
        testbed.lda(),
        testbed.batch_stride_a(),
        testbed.ptr_B(),
        testbed.ldb(),
        testbed.batch_stride_b(),
        testbed.beta(),
        testbed.ptr_C_initial(),
        testbed.ldc(),
        testbed.batch_stride_c(),
        testbed.ptr_experimental(),
        testbed.ldc(),
        testbed.batch_stride_c(),
        testbed.batch_count());

      dispatch_ptr->operator()();
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    if (CutlassDispatch::kRunCuBLAS) {
#if CUTLASS_ENABLE_CUBLAS
      if (testbed.verify_with_reference()) {
        result.disposition = Disposition::Passed;
      } else {
        result.disposition = Disposition::Incorrect;
      }
#endif
    }

    if (options.save_workspace(result.disposition == Disposition::Passed)) {
      write_problem(Provider::Cutlass, kernel_name);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
      result.disposition = Disposition::Failed;
    }

    // warmup launch
    dispatch_ptr->operator()();

    if (cudaDeviceSynchronize() != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    if (cudaEventRecord(events[0]) != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    for (int iter = 0; iter < options.iterations; ++iter) {
      dispatch_ptr->operator()();
    }

    if (cudaEventRecord(events[1]) != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    if (cudaEventSynchronize(events[1]) != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    float average_ms = 0;
    if (cudaEventElapsedTime(&average_ms, events[0], events[1]) != cudaSuccess) {
      result.disposition = Disposition::Failed;
      delete dispatch_ptr;
      return result;
    }

    result.runtime = double(average_ms) / double(options.iterations);
    result.gflops = testbed.GFLOPs_per_sec(result.runtime);

    if (result.disposition == Disposition::Unknown) {
      std::cout << "[\033[1;30mUnknown\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::NotRun) {
      std::cout << "[\033[1;33mNotRun\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::Passed) {
      std::cout << "[\033[1;32mPassed\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::Incorrect) {
      std::cout << "[\033[1;31mIncorrect\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::Failed) {
      std::cout << "[\033[1;31mFailed\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::NotVerified) {
      std::cout << "[\033[1;34mNotVerified\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    if (result.disposition == Disposition::Invalid) {
      std::cout << "[\033[1;36mInvalid\033[0m]: " << kernel_name
                << " with disposition: " << result.disposition << "\n";
    }
    delete dispatch_ptr;
    return result;
  }

  template <typename T, typename F>
  bool contains(T const &container, F const &val) {
    return std::find(container.begin(), container.end(), val) != container.end();
  }

  /// Executes all kernels for this problem size
  template <typename CutlassDispatch>
  std::vector<PerformanceResult<GemmProblem> > execute(GemmProblem const &problem) {

    // New problem size
    output.begin_problem();

    bool const tensor_op = !(CutlassDispatch::kThreadMultiplyAdd);
    cublasGemmAlgo_t algorithm = tensor_op ?
      CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

    testbed.resize(problem);

    std::vector<PerformanceResult<GemmProblem> > results;

      results.push_back(execute_cutlass<CutlassDispatch>(problem, algorithm));
    // cool-down period
    if (!options.dry_run) {
      pause(options.sleep_time);
    }

    return results;
  }

  /// Runs the test and collects performance for all results
  template <typename CutlassDispatch>
  void schmoo(Range const &M, Range const &N, Range const &K, Range const &batch_count) {
    for (int b = batch_count.start; b <= batch_count.end; b = batch_count.next(b)) {
      for (int m = M.start; m <= M.end; m = M.next(m)) {
        for (int n = N.start; n <= N.end; n = N.next(n)) {
          for (int k = K.start; k <= K.end; k = K.next(k)) {
            std::vector<PerformanceResult<GemmProblem> > results =
              execute<CutlassDispatch>(GemmProblem(m,
                n,
                k,
                CutlassDispatch::kLayoutA,
                CutlassDispatch::kLayoutB,
                config.alpha,
                config.beta,
                b));

            for (std::vector<PerformanceResult<GemmProblem> >::const_iterator it = results.begin();
              it != results.end();
              ++it) {
              output.append(*it);
            }
          }//k
        }//n
      }//m
    }//batch_count
  }

  /// Runs the test over the problem space and reports only the best performance
  template <typename CutlassDispatch>
  void peak(Range const &M, Range const &N, Range const &K) {
    typedef std::map<Provider::Kind, PerformanceResult<GemmProblem> > ProviderPerformanceMap;

    ProviderPerformanceMap max_perf;

    for (int m = M.start; m <= M.end; m += M.next(m)) {
      for (int n = N.start; n <= N.end; n += N.next(n)) {
        for (int k = K.start; k <= K.end; k += K.next(k)) {
          std::vector<PerformanceResult<GemmProblem> > results =
              execute<CutlassDispatch>(GemmProblem(m,
                                                   n,
                                                   k,
                                                   CutlassDispatch::kLayoutA,
                                                   CutlassDispatch::kLayoutB,
                                                   config.alpha,
                                                   config.beta));

          for (std::vector<PerformanceResult<GemmProblem> >::const_iterator it = results.begin();
               it != results.end();
               ++it) {
            /// Writes the output without appending it
            output.pretty_print(*it);

            if (it->disposition == Disposition::Passed) {
              /// Updates maximum performing kernel
              ProviderPerformanceMap::iterator max_perf_it = max_perf.find(it->provider);

              if (max_perf_it == max_perf.end()) {
                max_perf.insert(std::make_pair(it->provider, *it));
              } else if (max_perf_it->second.gflops < it->gflops) {
                max_perf_it->second = *it;
              }
            }
          }
        }
      }
    }

    Provider::Kind providers[] = {
      Provider::Cutlass,
      Provider::Invalid
    };
    for (int i = 0; providers[i] != Provider::Invalid; ++i) {
      ProviderPerformanceMap::const_iterator it = max_perf.find(providers[i]);
      if (it != max_perf.end()) {
        output.append(it->second);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatches to GEMM performance profiler
template <typename Dispatch, typename GemmProfiler>
int profile_gemm(TestbenchOutput<GemmProblem> &output,
                 std::string const &kernel,
                 TestbenchOptions const &options,
                 Config const &config,
                 std::string const &cutlass_algo = "") {
  if (config.kernel_enabled(kernel)) {
    GemmProfiler perf(output, kernel, cutlass_algo, options, config);
    if (options.peak_performance) {
      perf.template peak<Dispatch>(
          config.gemm_problem_range.M, config.gemm_problem_range.N, config.gemm_problem_range.K);
    } else {
      perf.template schmoo<Dispatch>(
          config.gemm_problem_range.M, config.gemm_problem_range.N, config.gemm_problem_range.K, config.gemm_problem_range.batch_count);
    }
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf
