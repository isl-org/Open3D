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

/** \file
    \brief CUTLASS Performance Tests
*/

#include <vector>
#include "tools/test/perf/performance_result.h"
#include "tools/test/perf/testbench_configs.h"
#include "tools/test/perf/testbench_options.h"
#include "tools/test/perf/testbench_output.h"

#include "tools/test/perf/cutlass_perf_test.h"

static std::vector<perf::GemmProfileFunc*> GemmProfileFuncs;

//
// Profiling entry points defined in corresponding .cu files
//
namespace perf {

void RegisterGemmProfileFunc(GemmProfileFunc * profileFunc) {
  GemmProfileFuncs.push_back(profileFunc);
}

}  // namespace perf

//
// Executes profiling functionality
//

template <typename Problem>
int profile(int (**functions)(perf::TestbenchOutput<Problem> &,
                              perf::TestbenchOptions const &,
                              perf::Config const &),
            perf::TestbenchOutput<Problem> &output,
            perf::TestbenchOptions options,
            int result) {
  perf::TestbenchConfigs test_configs(options);
  for (size_t j = 0; !result && j < test_configs.configs.size(); j++) {
    for (size_t i = 0; !result && functions[i] != 0; ++i) {
      result = (functions[i])(output, options, test_configs.configs[j]);
    }
  }
  return result;
}

/// Entry point to CUTLASS performance test
int main(int argc, const char **argv) {
  cutlass::CommandLine args(argc, argv);
  perf::TestbenchOptions options(args);

  if (args.check_cmd_line_flag("help")) {
    perf::TestbenchOptions::usage(std::cout);
    return 0;
  }

  if (args.check_cmd_line_flag("version")) {
    perf::TestbenchOptions::version(std::cout);
    std::cout << std::endl;
    return 0;
  }

  int result = 0;

      std::vector<perf::GemmProfileFunc*> profileFuncs = GemmProfileFuncs;
      profileFuncs.push_back(0); // Passing as array reference below, so need NULL termination.
      perf::TestbenchOutput<perf::GemmProblem> output_gemm(options);
      result = profile(&profileFuncs[0], output_gemm, options, result);
      return result;
}
