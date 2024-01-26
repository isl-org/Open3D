// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ProgressBar.h"

#include <chrono>
#include <thread>

#include <tbb/parallel_for.h>

#include "open3d/utility/Parallel.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(ProgressBar, ProgressBar) {
    int iterations = 1000;
    utility::ProgressBar progress_bar(iterations, "ProgressBar test: ", true);

    for (int i = 0; i < iterations; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        ++progress_bar;
    }
    EXPECT_EQ(iterations, static_cast<int>(progress_bar.GetCurrentCount()));
}

TEST(ProgressBar, OMPProgressBar) {
    int iterations = 1000;
    utility::OMPProgressBar progress_bar(iterations,
                                         "OMPProgressBar test: ", true);

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < iterations; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++progress_bar;
    }
    EXPECT_TRUE(static_cast<int>(progress_bar.GetCurrentCount()) >= iterations);
}

TEST(ProgressBar, TBBProgressBar) {
    int iterations = 1000;
    utility::TBBProgressBar progress_bar(
            iterations, "TBBProgressBar test: ", true);
    tbb::parallel_for(tbb::blocked_range<int>(
            0, iterations, utility::DefaultGrainSizeTBB()),
            [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ++progress_bar;
        }
    });
    EXPECT_TRUE(static_cast<int>(progress_bar.GetCurrentCount()) >= iterations);
}

}  // namespace tests
}  // namespace open3d
