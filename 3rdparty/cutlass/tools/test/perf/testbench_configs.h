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

#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <string>

#include "tools/test/perf/testbench_options.h"

namespace perf {

// Structure of configurations to run
struct Config {
  // Scalar value for GEMM
  double alpha;

  /// Scalar value for GEMM
  double beta;

  // kernel to run
  std::vector<std::string> kernels;

  /// Range of problem sizes for GEMM
  GemmProblemRange gemm_problem_range;
  
  // Reference GFLOPs
  double gflops_ref;

  // Reference Runtime
  double runtime_ref;

  // Reference Peak Throughput
  double peak_throughput_ref;

  // Returns true if the kernel name appears among the enabled kernels
  bool kernel_enabled(std::string const &kernel) const {
    typedef std::vector<std::string>::const_iterator kernel_iterator;

    for (kernel_iterator it = kernels.begin(); it != kernels.end(); ++it) {
      if (kernel.find(*it) != std::string::npos) {
        return true;
      }
    }

    return false;
  }
};

// Class to set the configurations to run
struct TestbenchConfigs {
  //
  // Data members
  //

  // Vector of configurations to run
  std::vector<perf::Config> configs;

  // Options to test environment
  TestbenchOptions options;

  // Input CSV file to read (if applicable)
  std::ifstream threshold_file;

  //
  // Methods
  //

  // Determines the configurations to run from the threshold file
  void configs_from_file() {
    // Set the values of kernels, M, N, K and beta based off of values read from CSVs
    threshold_file.open(options.threshold_filename.c_str());
    if (threshold_file.is_open()) {
      std::string line;
      int provider_idx = -1;
      int kernel_idx = -1;
      int beta_idx = -1;
      int m_idx = -1;
      int n_idx = -1;
      int k_idx = -1;
      int gflops_idx = -1;
      int runtime_idx = -1;
      int peak_throughput_idx = -1;

      // Read the header and get the indices of the columns
      if (getline(threshold_file, line)) {
        char delim = ',';
        size_t s_idx = 0;
        size_t d_idx = std::string::npos;
        int idx = 0;
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        while (s_idx < line.size()) {
          d_idx = line.find_first_of(delim, s_idx);
          size_t end_idx = (d_idx != std::string::npos ? d_idx : line.size());
          std::string item = line.substr(s_idx, end_idx - s_idx);
          if (item.compare("Provider") == 0) provider_idx = idx;
          if (item.compare("Kernel") == 0) kernel_idx = idx;
          if (item.compare("Beta") == 0) beta_idx = idx;
          if (item.compare("M") == 0) m_idx = idx;
          if (item.compare("N") == 0) { 
            n_idx = idx;
          }
          if (item.compare("K") == 0) { 
            k_idx = idx; 
          }
          if (item.compare("GFLOPs") == 0) gflops_idx = idx;
          if (item.compare("Runtime") == 0) runtime_idx = idx;
          if (item.compare("SOL") == 0) peak_throughput_idx = idx;
          s_idx = end_idx + 1;  // For comma
          idx++;
        }
      }

      while (getline(threshold_file, line)) {
        char delim = ',';
        size_t s_idx = 0;
        size_t d_idx = std::string::npos;
        std::vector<std::string> tokens;
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        while (s_idx < line.size()) {
          d_idx = line.find_first_of(delim, s_idx);
          size_t end_idx = (d_idx != std::string::npos ? d_idx : line.size());
          std::string item = line.substr(s_idx, end_idx - s_idx);
          tokens.push_back(item);
          s_idx = end_idx + 1;  // For comma
        }
        if (tokens[provider_idx].compare("Cutlass") == 0) {
          // Create a new config
          Config config = Config();
          config.alpha = options.alpha;
          config.beta = strtod(tokens[beta_idx].c_str(), NULL);
          config.kernels.push_back(tokens[kernel_idx]);
          config.gemm_problem_range.M = Range(tokens[m_idx]);
          config.gemm_problem_range.N = Range(tokens[n_idx]);
          config.gemm_problem_range.K = Range(tokens[k_idx]);
          config.gflops_ref = strtod(tokens[gflops_idx].c_str(), NULL);
          config.runtime_ref = strtod(tokens[runtime_idx].c_str(), NULL);
          config.peak_throughput_ref = strtod(tokens[peak_throughput_idx].c_str(), NULL);
          configs.push_back(config);
        }
      }
    } else {  // !threshold_file.is_open()
      std::cout << "ERROR: Could not open threshold file " << options.threshold_filename << "\n";
    }
  }

  // Determines the configurations to run from the command line arguments
  void configs_from_args() {
    Config config = Config();
    config.alpha = options.alpha;
    config.beta = options.beta;
    for (int i = 0; i < options.kernels.size(); i++) {
      config.kernels.push_back(options.kernels[i]);
    }
    config.gemm_problem_range = options.gemm_problem_range;
    configs.push_back(config);
  }

  // Constructor
  TestbenchConfigs(TestbenchOptions const &_options) : options(_options) {
    if (!options.threshold_filename.empty()) {
      configs_from_file();
    } else {
      configs_from_args();
    }
  }
};

}  // namespace perf
