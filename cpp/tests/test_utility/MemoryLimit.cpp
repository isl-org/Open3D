// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "tests/test_utility/MemoryLimit.h"

#include <cstdlib>
#include <iostream>

#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

namespace {

/// return value of environment variable as a number, or -1 if it is not set
int GetEnvNumber(const std::string &env) {
    const char *v = getenv(env.c_str());
    if (!v) return -1;
    return atol(v);
}

void SkipCurrentTest() { GTEST_SKIP_("Hello world"); }

}  // namespace

using open3d::core::Device;

bool OverMemoryLimit(std::string test_name,
                     int cpu_mb,
                     int gpu_mb,
                     Device device) {
    static const int cpu_limit = GetEnvNumber("TEST_MAX_CPU_MEMORY_MB");
    static const int gpu_limit = GetEnvNumber("TEST_MAX_GPU_MEMORY_MB");
    static const bool report_memory_limits =
            (GetEnvNumber("OPEN3D_TEST_REPORT_MEMORY_LIMITS") == 1);
    bool over_cpu_limit = (cpu_limit > 0 && cpu_mb > cpu_limit);
    bool over_gpu_limit = ((device.GetType() == Device::DeviceType::CUDA) &&
                           (gpu_limit > 0 && gpu_mb > gpu_limit));
    bool skip = over_cpu_limit || over_gpu_limit;
    if (report_memory_limits) {
        // print out details on stderr so it can be captured along with memory
        // allocation info
        std::cerr << fmt::format(
                "Open3dTestMemoryLimits test_name {} cpu_mb {} gpu_mb {} skip "
                "{} device {} cpu_limit {} gpu_limit {} over_cpu_limit {} "
                "over_gpu_limit {}\n",
                test_name, cpu_mb, gpu_mb, skip, device.ToString(), cpu_limit,
                gpu_limit, over_cpu_limit, over_gpu_limit);
    }

    if (skip) {
        utility::LogWarning(
                "Skipping test {} (device {}), memory utilization (cpu: {} mb, "
                "GPU: {} mb) exceed max test memory (cpu: {} mb, gpu: {} mb)",
                test_name, device.ToString(), cpu_mb, gpu_mb, cpu_limit,
                gpu_limit);
        SkipCurrentTest();
    }
    return skip;
}

}  // namespace tests
}  // namespace open3d
