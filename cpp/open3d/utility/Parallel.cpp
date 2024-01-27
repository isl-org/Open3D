// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Parallel.h"

#include <tbb/task_arena.h>


namespace open3d {
namespace utility {

int EstimateMaxThreads() {
    return tbb::this_task_arena::max_concurrency();
}

std::size_t& DefaultGrainSizeTBB() noexcept {
    static std::size_t GrainSize = 256;
    return GrainSize;
}

std::size_t& DefaultGrainSizeTBB2D() noexcept {
    static std::size_t GrainSize = 32;
    return GrainSize;
}

}  // namespace utility
}  // namespace open3d
