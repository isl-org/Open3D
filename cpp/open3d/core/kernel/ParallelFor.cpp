// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/kernel/ParallelFor.h"

#include <cstdint>
#include <functional>
#include <thread>

#include "open3d/core/kernel/ParallelUtil.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

// Value taken from PyTorch's at::internal::GRAIN_SIZE. The value is chosen
// heuristically.
static constexpr int64_t DEFAULT_MIN_PARALLEL_SIZE = 32768;

void ParallelFor(int64_t num_jobs, const std::function<void(int64_t)>& f) {
    ParallelFor(0, num_jobs, DEFAULT_MIN_PARALLEL_SIZE, f);
}

void ParallelFor(int64_t start,
                 int64_t end,
                 const std::function<void(int64_t)>& f) {
    ParallelFor(start, end, DEFAULT_MIN_PARALLEL_SIZE, f);
}

void ParallelFor(int64_t start,
                 int64_t end,
                 int64_t min_parallel_size,
                 const std::function<void(int64_t)>& f) {
    // unsigned int num_cpu = std::thread::hardware_concurrency();
    // utility::LogInfo("num_cpu = {}.", num_cpu);

    if (min_parallel_size <= 0) {
        utility::LogError("min_parallel_size must be > 0, but got {}.",
                          min_parallel_size);
    }

    // It's also possible to use `#pragma omp parallel for if (xxx)`.
    if (end - start <= min_parallel_size || GetMaxThreads() == 1 ||
        InParallel()) {
        for (int64_t i = start; i < end; i++) {
            f(i);
        }
    } else {
#pragma omp parallel for
        for (int64_t i = start; i < end; i++) {
            f(i);
        }
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
