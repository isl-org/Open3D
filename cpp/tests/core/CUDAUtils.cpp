// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef BUILD_CUDA_MODULE

#include "open3d/core/CUDAUtils.h"

#include <sstream>
#include <thread>
#include <vector>

#include "tests/Tests.h"

// #include <fmt/std.h>   // fmt version >=9, else:
namespace {
// Use std::ostringstream to get string representation.
template <typename T>
std::string to_str(T var) {
    std::ostringstream ss;
    ss << var;
    return ss.str();
}
}  // namespace

namespace open3d {
namespace tests {

TEST(CUDAUtils, InitState) {
    const int device_count = core::cuda::DeviceCount();
    const core::CUDAState& cuda_state = core::CUDAState::GetInstance();
    utility::LogInfo("Number of CUDA devices: {}", device_count);
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            utility::LogInfo("P2PEnabled {}->{}: {}", i, j,
                             cuda_state.IsP2PEnabled(i, j));
        }
    }
}

void CheckScopedStreamManually() {
    int current_device = core::cuda::GetDevice();

    ASSERT_EQ(core::CUDAStream::GetInstance().Get(),
              core::CUDAStream::Default().Get());
    ASSERT_EQ(core::cuda::GetDevice(), current_device);

    core::CUDAStream stream = core::CUDAStream::CreateNew();

    {
        core::CUDAScopedStream scoped_stream(stream);

        ASSERT_EQ(core::CUDAStream::GetInstance().Get(), stream.Get());
        ASSERT_FALSE(core::CUDAStream::GetInstance().IsDefaultStream());
        ASSERT_EQ(core::cuda::GetDevice(), current_device);
    }

    stream.Destroy();

    ASSERT_TRUE(core::CUDAStream::GetInstance().IsDefaultStream());
    ASSERT_EQ(core::cuda::GetDevice(), current_device);
}

void CheckScopedStreamAutomatically() {
    int current_device = core::cuda::GetDevice();

    ASSERT_EQ(core::CUDAStream::GetInstance().Get(),
              core::CUDAStream::Default().Get());
    ASSERT_EQ(core::cuda::GetDevice(), current_device);

    {
        core::CUDAScopedStream scoped_stream(core::CUDAStream::CreateNew(),
                                             true);

        ASSERT_NE(core::CUDAStream::GetInstance().Get(),
                  core::CUDAStream::Default().Get());
        ASSERT_EQ(core::cuda::GetDevice(), current_device);
    }

    ASSERT_EQ(core::CUDAStream::GetInstance().Get(),
              core::CUDAStream::Default().Get());
    ASSERT_EQ(core::cuda::GetDevice(), current_device);
}

void CheckScopedStreamMultiThreaded(const std::function<void()>& func) {
    std::vector<std::thread> threads;

    // NVCC does not like capturing const int by reference on Windows, use int.
    int kIterations = 100000;
    int kThreads = 8;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&kIterations, &func]() {
            utility::LogDebug("Starting thread with ID {}",
                              to_str(std::this_thread::get_id()));
            for (int i = 0; i < kIterations; ++i) {
                func();
            }
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            utility::LogDebug("Joining thread with ID {}",
                              to_str(thread.get_id()));
            thread.join();
        }
    }
}

TEST(CUDAUtils, ScopedStreamManually) { CheckScopedStreamManually(); }

TEST(CUDAUtils, ScopedStreamManuallyMultiThreaded) {
    CheckScopedStreamMultiThreaded(&CheckScopedStreamManually);
}

TEST(CUDAUtils, ScopedStreamAutomatically) { CheckScopedStreamAutomatically(); }

TEST(CUDAUtils, ScopedStreamAutomaticallyMultiThreaded) {
    CheckScopedStreamMultiThreaded(&CheckScopedStreamAutomatically);
}

}  // namespace tests
}  // namespace open3d

#endif
