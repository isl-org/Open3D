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

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdint.h>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "tools/util/command_line.h"
#include "tools/util/distribution.h"
#include "tools/test/perf/provider.h"

namespace perf {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Range of problem sizes
struct Range {

  enum Operator {
    Add,
    Multiply
  };

  //
  // Data members
  //

  int start;
  int end;
  int increment;
  Operator increment_op;

  //
  // Methods
  //

  Range(int _start = 0) : start(_start), end(_start), increment(1), increment_op(Add) {}

  Range(int _start, int _end, int _increment = 1, Operator _op = Add)
      : start(_start), end(_end), increment(_increment), increment_op(_op) {}

  Range(std::string _start) : increment(1), increment_op(Add) {
    start = end = (int)strtol(_start.c_str(), NULL, 10);
  }


  /// Returns the next item in series
  int next(int val) const {
    switch (increment_op) {
      case Add: val += increment; break;
      case Multiply: val *= increment; break;
      default: val = end; break;
    }
    return val;
  }

  void import_from_strings(const std::vector<std::string>& values) {
    if (values.size() > 0) {
      std::stringstream ss;
      ss << values.at(0);
      ss >> start;
    }

    if (values.size() > 1) {
      std::stringstream ss;
      ss << values.at(1);
      ss >> end;
    } else {
      end = start;
    }

    if (values.size() > 2 && !values.at(2).empty()) {
      std::stringstream ss;

      char first = values.at(2).at(0);
      if (first == '*' || first == '+') {
        ss << values.at(2).substr(1);
        switch (first) {
        case '*': increment_op = Multiply; break;
        case '+': increment_op = Add; break;
        default: break;
        }
      }
      else {
        ss << values.at(2);
      }
      ss >> increment;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a space of problem sizes
struct GemmProblemRange {
 public:
  /// Range of sizes in GEMM M dimension
  Range M;

  /// Range of sizes in GEMM N dimension
  Range N;

  /// Range of sizes in GEMM K dimension
  Range K;

  /// Range of sizes in batch dimeion
  Range batch_count;

  //
  // Methods
  //

  /// Constructor to define a space of probelm sizes
  GemmProblemRange(Range _M = Range(256), Range _N = Range(256), Range _K = Range(256), Range _batch_count = Range(1))
      : M(_M), N(_N), K(_K), batch_count(_batch_count) {}

  /// Parses a command line argument as a Range object
  static void get_range(Range &range,
                        cutlass::CommandLine const &args,
                        std::string const &arg,
                        Range const &_default = Range(256)) {
    range = Range(0, 0, 1);

    if (args.check_cmd_line_flag(arg.c_str())) {
      std::vector<std::string> values;
      args.get_cmd_line_arguments(arg.c_str(), values, ':');

      range.import_from_strings(values);
    } else {
      range = _default;
    }
  }

  /// Initializes the GEMM problem size from command line arguments
  GemmProblemRange(cutlass::CommandLine const &args) {
    get_range(M, args, "m", Range(10240));
    get_range(N, args, "n", Range(4096));
    get_range(K, args, "k", Range(4096));
    get_range(batch_count, args, "batch", Range(1));
  }
};

/// Defines a vector of string pairs
typedef std::vector<std::pair<std::string, std::string> > KeyValueVector;

/// Defines a const iterator to a KeyValueVector
typedef KeyValueVector::const_iterator KeyValueIterator;

/// Structure captures the initial configuration of matrices
struct InitialDistribution {
  /// Distribution of A matrix operand
  cutlass::Distribution dist_A;

  /// Distribution of B matrix operand
  cutlass::Distribution dist_B;

  /// cutlass::Distribution of C matrix operand
  cutlass::Distribution dist_C;

  /// Seed for random number generation
  int64_t seed;

  //
  // Static function members
  //

  /// Gets the initial distribution
  static void get_distribution(cutlass::CommandLine const &args,
                               std::string const &arg,
                               cutlass::Distribution &dist) {
    struct {
      const char *label;
      cutlass::Distribution::Kind kind;
    } distribution_kinds[] = {{"uniform", cutlass::Distribution::Uniform},
                              {"gaussian", cutlass::Distribution::Gaussian},
                              {"linear", cutlass::Distribution::Linear},
                              {"identity", cutlass::Distribution::Identity},
                              {0, cutlass::Distribution::Invalid}};

    struct {
      char const *label;
      double *member;
    } members[] = {{"min", &dist.uniform.min},
                   {"max", &dist.uniform.max},
                   {"mean", &dist.gaussian.mean},
                   {"stddev", &dist.gaussian.stddev},
                   {"offset", &dist.linear.offset},
                   {"delta_row", &dist.linear.delta_row},
                   {"delta_column", &dist.linear.delta_column},
                   {0, 0}};

    KeyValueVector values;
    args.get_cmd_line_argument_pairs(arg.c_str(), values);

    // The parser expects the first token to be a string identifying the distribution type.
    KeyValueIterator it = values.begin();
    if (it != values.end()) {
      for (int i = 0; distribution_kinds[i].label; ++i) {
        if (it->first == distribution_kinds[i].label) {
          dist.kind = distribution_kinds[i].kind;
          break;
        }
      }
      ++it;
    }

    // Subsequent key-value pairs update the named field of the distribution struct.
    for (; it != values.end(); ++it) {
      // Integer scaling factor - if < 0, no integer rounding is performed.
      if (it->first == "scale" && !it->second.empty()) {
        std::stringstream ss;
        ss << it->second;
        ss >> dist.int_scale;
        continue;  // next token
      }

      // Casts as integer without scaling
      if (it->first == "integer") {
        dist.int_scale = 0;
        continue;  // next token
      }

      // Casts as integer without scaling
      if (it->first == "integer") {
        dist.int_scale = 0;
        continue;  // next token
      }

      // initialize other members
      for (int m = 0; members[m].label; ++m) {
        if (it->first == members[m].label && !it->second.empty()) {
          std::stringstream ss;
          ss << it->second;
          ss >> *(members[m].member);
        }
      }
    }
  }

  //
  // Methods
  //

  /// Basic uniform random distribution
  InitialDistribution(int64_t _seed = 700) : seed(_seed) {
    dist_A.set_uniform(-4, 4);
    dist_B.set_uniform(-4, 4);
    dist_C.set_uniform(-4, 4);
  }

  /// Extracts initial distribution from command line arguments
  InitialDistribution(cutlass::CommandLine const &args) {
    // Set initial values
    seed = 700;
    dist_A.set_uniform(-4, 4);
    dist_B.set_uniform(-4, 4);
    dist_C.set_uniform(-4, 4);

    // Update with command line arguments
    args.get_cmd_line_argument("seed", seed, seed);

    // Update all distributions at once
    cutlass::Distribution dist_all;
    if (args.check_cmd_line_flag("dist")) {
      get_distribution(args, "dist", dist_all);
      dist_A = dist_all;
      dist_B = dist_all;
      dist_C = dist_all;
    }

    get_distribution(args, "dist_A", dist_A);
    get_distribution(args, "dist_B", dist_B);
    get_distribution(args, "dist_C", dist_C);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines how to execute the benchmarks
struct ExecutionMode {
  enum Kind { Profile, Verify, Single, Invalid };

  static std::string to_string(Kind kind) {
    switch (kind) {
      case Profile:
        return "profile";
      case Verify:
        return "verify";
      case Single:
        return "single";
      default:
        return "invalid";
    }
  }

  static Kind from_string(std::string const &str) {
    if (str == "profile") return Profile;
    if (str == "verify") return Verify;
    if (str == "single") return Single;
    return Profile;
  }
};

/// Indicates when the workspace is saved
struct WorkspaceSaveMode {
  enum Kind { Never, Incorrect, Always };

  static std::string to_string(Kind kind) {
    switch (kind) {
      case Never:
        return "never";
      case Incorrect:
        return "incorrect";
      case Always:
        return "always";
      default:
        return "incorrect";
    }
  }

  static Kind from_string(std::string const &str) {
    if (str == "never") return Never;
    if (str == "incorrect") return Incorrect;
    if (str == "always") return Always;
    return Incorrect;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Class holding testbench command line options
struct TestbenchOptions {
  //
  // Data members
  //

  /// Describes the random initial state of the input matrices
  InitialDistribution initial_distribution;

  // Path to output file name
  std::string output_filename;

  // Path to input file name
  std::string threshold_filename;

  /// If true, output is appended
  bool append;

  /// Number of iterations
  int iterations;

  /// Defines how to run the benchmark
  ExecutionMode::Kind execution_mode;

  /// Indicates when the workspace is saved
  WorkspaceSaveMode::Kind save_workspace_mode;

  /// Properties of CUDA device
  cudaDeviceProp device_properties;

  /// Enabled kernel names
  std::vector<std::string> kernels;

  /// Scalar value for GEMM
  double alpha;

  /// Scalar value for GEMM
  double beta;

  /// Range of GEMM problem sizes
  GemmProblemRange gemm_problem_range;

  /// If true, kernels are not executed, and no sleep waits are inserted
  bool dry_run;

  /// Tags to describe the profiler output
  KeyValueVector pivot_tags;

  /// If enabled, only the peak performance for a given kernel is reported
  bool peak_performance;

  /// Performance Degradatiom Margin before flagging as test failure
  double perf_margin;

  /// Cool-down period
  int sleep_time;

  //
  // Methods
  //

  /// Constructs the testbench from tags
  TestbenchOptions(cutlass::CommandLine const &args)
      : initial_distribution(args),
        execution_mode(ExecutionMode::Profile),
        save_workspace_mode(WorkspaceSaveMode::Never),
        gemm_problem_range(args),
        dry_run(false),
        sleep_time(1) {

    // Set the CUDA device and/or specify clock rate
    configure_cuda_device(args);

    // fetch command line arguments
    args.get_cmd_line_argument("iterations", iterations, 25);
    args.get_cmd_line_argument("append", append, false);
    args.get_cmd_line_argument("output", output_filename);
    args.get_cmd_line_argument("threshold", threshold_filename);
    args.get_cmd_line_argument("alpha", alpha, 1.0);
    args.get_cmd_line_argument("beta", beta, 0.0);
    args.get_cmd_line_argument("peak", peak_performance, false);
    args.get_cmd_line_argument_pairs("tags", pivot_tags);
    args.get_cmd_line_argument("perf-margin", perf_margin, 0.97);
    args.get_cmd_line_argument("dry-run", dry_run, false);
    args.get_cmd_line_argument("sleep-time", sleep_time, 1);

    if (args.check_cmd_line_flag("execution-mode")) {
      std::string str;
      args.get_cmd_line_argument("execution-mode", str);
      execution_mode = ExecutionMode::from_string(str);
    }

    if (args.check_cmd_line_flag("save-workspace")) {
      std::string str;
      args.get_cmd_line_argument("save-workspace", str);
      save_workspace_mode = WorkspaceSaveMode::from_string(str);
    }

    if (args.check_cmd_line_flag("execution-mode")) {
      std::string str;
      args.get_cmd_line_argument("execution-mode", str);
      execution_mode = ExecutionMode::from_string(str);
    }

    if (args.check_cmd_line_flag("save-workspace")) {
      std::string str;
      args.get_cmd_line_argument("save-workspace", str);
      save_workspace_mode = WorkspaceSaveMode::from_string(str);
    }

    // query for enabled kernels or enable all of them
    if (args.check_cmd_line_flag("kernels")) {
      args.get_cmd_line_arguments("kernels", kernels, ',');
    } else {
      char const *gemms[] = {
        "sgemm",
        "dgemm",
        "hgemm",
        "igemm",
        "wmma_gemm",
        "wmma_gemm_f16",
        "s884gemm",
        "h884gemm",
        "wmma_binary_gemm",
        "wmma_integer_gemm",
        0
      };
      char const *layouts[] = {"nn", "nt", "tn", "tt", 0};
      for (int i = 0; gemms[i]; ++i) {
        for (int j = 0; layouts[j]; ++j) {
          if ((
               std::string(gemms[i]).compare("wmma_binary_gemm") == 0 ||
               std::string(gemms[i]).compare("wmma_integer_gemm") == 0)
               && std::string(layouts[j]).compare("tn") != 0) {
            continue;
          }
          kernels.push_back(std::string(gemms[i]) + "_" + layouts[j]);
        }
      }
      
    }
  }

  void configure_cuda_device(cutlass::CommandLine const &args) {
    int device_id = 0;
    args.get_cmd_line_argument("device", device_id, 0);

    cudaError_t result;
    result = cudaGetDeviceProperties(&device_properties, device_id);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed for given device.");
    }
    result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice() failed for given device.");
    }

    // Get the clock rate (specified in cmd line in MHz)
    if (args.check_cmd_line_flag("clock")) {
      args.get_cmd_line_argument("clock", device_properties.clockRate);
      device_properties.clockRate *= 1000;
    }
  }

  /// Returns true if the kernel name appears among the enabled kernels
  bool kernel_enabled(std::string const &kernel) const {
    typedef std::vector<std::string>::const_iterator kernel_iterator;

    for (kernel_iterator it = kernels.begin(); it != kernels.end(); ++it) {
      if (kernel.find(*it) != std::string::npos) {
        return true;
      }
    }

    return false;
  }

  /// Given the disposition of a GEMM problem, returns true if the results should
  /// be saved to the file system.
  bool save_workspace(bool correct) const {
    if (save_workspace_mode == WorkspaceSaveMode::Always ||
        (save_workspace_mode == WorkspaceSaveMode::Incorrect && !correct)) {
      return true;
    }
    return false;
  }

  /// Returns true if the selected device can satisfy the given compute capability
  bool compute_capability(int major, int minor) const {
    return (device_properties.major > major ||
      (device_properties.major == major && device_properties.minor >= minor));
  }

  /// Requires an exact match of compute capability
  bool compute_capability_exact(int major, int minor) const {
    return major == device_properties.major && minor == device_properties.minor;
  }

  /// Prints version
  static void version(std::ostream &out) {
    out << "CUTLASS " << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "." << CUTLASS_PATCH
        << " built on " << __DATE__ << " at " << __TIME__;
  }

  /// Prints the usage statement
  static void usage(std::ostream &out) {
    out << "cutlass_perf_test [options]\n\n"

        << "  --help\n"

        << "  --append=<true|false*>                        "
        << "    If true, appends output to existing CSV file. If false, overwrites.\n"

        << "  --alpha=<alpha>                               "
        << "    Value for alpha to be used in GEMM experiments\n"

        << "  --beta=<beta>                                 "
        << "    Value for beta to be used in GEMM experiments\n"

        << "  --device=<int>                                "
        << "    Specifies the CUDA device to use. Default is device 0.\n"

        << "  --clock=<MHz>                                 "
        << "    Specifies the SM clock rate in MHz.\n"

        << "  --dist-{A,B,C}=<distribution>                 "
        << "    Describes the random distribution of each of the input matrix operands.\n"

        << "  --dry-run=<bool>                              "
        << "    If true, kernels are not executed and sleep is not inserted.\n"

        << "  --execution-mode=<mode>                       "
        << "    Specifies execution mode: profile, verify, single\n"

        << "  --output=<filename.csv>                       "
        << "    Writes summary of profiling to specified .csv file\n"

        << "  --threshold=<filename.csv>                    "
        << "    Reads previous output summary and re-executes the same configurations.\n"

        << "  --iterations=<timing iterations>              "
        << "    maximum number of iterations to execute when profiling\n"

        << "  --m=<height>[:max height[:step]]              "
        << "    Height of GEMM problem (number of rows of C). May specify a range with optional "
           "step size.\n"

        << "  --n=<width>[:max width[:step]] (GEMM-specific)"
        << "    Width of GEMM problem (number of columns of C). May specify a range with optional "
           "step size.\n"

        << "  --k=<depth>[:max depth[:step]] (GEMM-specific)"
        << "    Size of inner dimension of A and B. May specify a range with optional step size.\n"
        << "  --batch=<batch>                               "
        << "    Number of batches for a batched gemm.\n"

        << "  --kernels=<{s|d|h|i|wmma_|wmma_binary_|wmma_integer_}gemm_{nn,nt,tn,tt}>\n"
        << "                                                "
        << "    Select GEMM datatype and layout to use for tests\n"

        << "  --peak=<bool>                                 "
        << "    If true, only reports peak performance per kernel after profiling specified "
           "problem space.\n"

        << "  --perf-margin=<perf-margin>                   "
        << "    Allowable performance degradation before flagging test as failure (e.g. 3% slowdown"
           " = 0.97).\n"

        << "  --save-workspace={*never,incorrect,always}    "
        << "    Specifies when to save the GEMM inputs and results to the filesystem.\n"

        << "  --seed=<seed>                                 "
        << "    Random seed used by the random number generator in initializing input matrices.\n"

        << "  --tags=<column:tag,...>                       "
        << "    Inserts leading columns in output table and uniform values for each column. Useful "
           "for generating pivot tables.\n"

        << "  --sleep-time=<second>                         "
        << "    Sleep period between profiling kernels to cool down the device.\n"

        << "  --version                                     "
        << "    ";

    version(out);

    out << "\n\n";

    out << "\n\n"
        << "Example usage:\n\n"

        << "# Runs one problem size for all GEMM kernels\n"
        << "./tools/test/perf/cutlass_perf_test --m=10240 --n=1024 --k=1024\n\n"

        << "# Varies GEMM K dimension for SGEMM and IGEMM with column-major multiplicands\n"
        << "./tools/test/perf/cutlass_perf_test --m=10240 --n=4096 --k=1024:8192:128 "
           "--kernels=sgemm_nn,igemm_nn\n\n"
        << " # Executes GEMM kernel on Volta Tensor Cores\n"
        << " $ ./tools/test/perf/cutlass_perf_test --kernels=s884gemm_nt\n\n"
        << std::flush;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace perf
