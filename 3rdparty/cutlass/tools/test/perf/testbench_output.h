/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without specific prior written
 *     permission.
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

#include "tools/test/perf/performance_result.h"
#include "tools/test/perf/testbench_options.h"
#include "tools/util/command_line.h"

namespace perf {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wraps an output stream and constructs a comma-separated value table of results
template <typename Problem>
class TestbenchOutput {
 public:
  /// Options to test environment
  TestbenchOptions const &options;

  /// Possibly open output file name
  std::ofstream output_file;

  /// Pointer to either &std::cout or output_file
  std::ostream *output_ptr;

  /// if true, output is also printed to std::cout in human readable form
  bool buffer_csv_output;

  /// Vector holding performance results
  std::vector<PerformanceResult<Problem> > buffered_perf_results;

 private:
  /// Opens the output file and updates output_ptr
  void initialize_output_file() {
    std::ifstream test_file(options.output_filename.c_str());
    if (options.append && test_file.good()) {
      output_file.open(options.output_filename.c_str(), std::ios::app);
    } else {
      output_file.open(options.output_filename.c_str());
      output_file << header() << std::endl;
    }
    output_ptr = &output_file;
  }

 public:
  /// Emits the header to the output table
  std::string header() {
    std::stringstream ss;

    // pivot tags
    for (KeyValueIterator tag_it = options.pivot_tags.begin(); tag_it != options.pivot_tags.end();
         ++tag_it) {
      ss << tag_it->first << ",";
    }

    // performance result header
    ss << PerformanceResult<Problem>::header();

    return ss.str();
  }

  /// Constructs a TestbenchoutOutput object from command line options
  TestbenchOutput(TestbenchOptions const &_options) : options(_options), buffer_csv_output(true) {
    if (!options.output_filename.empty()) {
      initialize_output_file();
      buffer_csv_output = false;
    } else {
      output_ptr = &std::cout;
    }
  }

  /// Writes output to CSV
  ~TestbenchOutput() {
    if (buffered_perf_results.size() != 0) {
      std::cout << std::endl;
      if (buffer_csv_output) {
        out() << "\n\n" << header() << std::endl;
        for (typename std::vector<PerformanceResult<Problem> >::const_iterator it =
                 buffered_perf_results.begin();
             it != buffered_perf_results.end();
             ++it) {
          write_csv(*it);
        }
      }
        std::cout << "\n[\033[1;32mPASSED\033[0m]";
        if (!options.threshold_filename.empty()) {
          std::cout << " - Performance Test Successful" << std::endl;
        } else {
          std::cout << std::endl;
        }
    }
  }

  /// Returns a reference to an std::ostream instance for writing
  std::ostream &out() { return *output_ptr; }

  /// Called to indicate a new problem will be output
  TestbenchOutput &begin_problem() {
    std::cout << "\n============================================================================\n";

    for (KeyValueIterator tag_it = options.pivot_tags.begin(); tag_it != options.pivot_tags.end();
         ++tag_it) {
      std::cout << tag_it->first << ": " << tag_it->second << std::endl;
    }

    return *this;
  }

  /// Writes a performance result to CSV output
  TestbenchOutput &write_csv(PerformanceResult<Problem> const &result) {
    // pivot tags
    for (KeyValueIterator tag_it = options.pivot_tags.begin(); tag_it != options.pivot_tags.end();
         ++tag_it) {
      out() << tag_it->second << ",";
    }

    out() << result << std::endl;
    return *this;
  }

  /// Prints the output without appending it for CSV writing
  TestbenchOutput &pretty_print(PerformanceResult<Problem> const &result) {
    result.pretty_print(std::cout) << std::endl;

    return *this;
  }

  /// Emits the result as output
  TestbenchOutput &append(PerformanceResult<Problem> const &result) {
    if (buffer_csv_output) {
      buffered_perf_results.push_back(result);
    } else {
      write_csv(result);
      buffered_perf_results.push_back(result);
    }

    pretty_print(result);

    return *this;
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf
