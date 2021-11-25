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

#include "open3d/utility/ProgressBar.h"

#include <chrono>
#include <thread>

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

}  // namespace tests
}  // namespace open3d
