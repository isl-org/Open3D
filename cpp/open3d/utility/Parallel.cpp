// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Parallel.h"

#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <memory>
#include <mutex>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

int EstimateMaxThreads() {
    // The arena concurrency does not reflect a global_control limit (see
    // SetMaxThreads), so take the smaller of the two.
    const std::size_t global_limit = tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);
    const int arena_limit = tbb::this_task_arena::max_concurrency();
    return std::min(static_cast<int>(global_limit), arena_limit);
}

std::size_t& DefaultGrainSizeTBB() noexcept {
    // Used for 1D loops whose body is expensive (a KDTree query, an
    // Eigen solve, ...), i.e. roughly a microsecond or more per item.
    // A range shorter than the grain size is never split, so the grain also
    // sets the smallest problem that can use all cores: with 64, a loop is
    // fully parallel from ~64*num_threads items. Measured on a 20-core
    // machine, 64 matches 256 for large inputs and is up to ~2x faster for
    // inputs of a few thousand items, where 256 leaves most cores idle.
    static std::size_t GrainSize = 64;
    return GrainSize;
}

std::size_t& DefaultGrainSizeTBB2D() noexcept {
    // Used for 2D/3D blocked ranges and for element-wise kernels
    // (core::ParallelFor), where each item is only a few flops. These loops
    // are usually memory bound, so the grain only needs to be large enough to
    // amortize the task overhead.
    static std::size_t GrainSize = 32;
    return GrainSize;
}

void SetMaxThreads(int num_threads) {
    // A tbb::global_control is only in effect while it is alive, so the object
    // is kept in a static and replaced on each call. The mutex guards against
    // concurrent replacement; the limit itself applies process-wide.
    static std::mutex mutex;
    static std::unique_ptr<tbb::global_control> limit;

    if (num_threads < 0) {
        LogError("num_threads must be >= 0, but got {}.", num_threads);
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (num_threads == 0) {
        limit.reset();
    } else {
        limit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                static_cast<std::size_t>(num_threads));
    }
}

}  // namespace utility
}  // namespace open3d
