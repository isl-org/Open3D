// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
    static std::size_t GrainSize = 256;
    return GrainSize;
}

std::size_t& DefaultGrainSizeTBB2D() noexcept {
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
