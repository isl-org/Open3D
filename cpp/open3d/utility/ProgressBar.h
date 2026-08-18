// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <string>

namespace open3d {
namespace utility {

/// Thread-safe progress bar. Progress updates use relaxed atomics, so callers
/// do not need to hold a mutex while incrementing it from parallel work.
class ProgressBar {
public:
    ProgressBar(std::size_t expected_count,
                std::string progress_info,
                bool active = false);
    ProgressBar(const ProgressBar& other);
    ProgressBar& operator=(const ProgressBar& other);
    void Reset(std::size_t expected_count,
               std::string progress_info,
               bool active);
    inline ProgressBar& operator++() { return *this += 1; };
    virtual ProgressBar& operator+=(std::size_t n);
    void SetCurrentCount(size_t n);
    void UpdateBar();
    std::size_t GetCurrentCount() const;
    virtual ~ProgressBar() = default;

protected:
    static constexpr size_t resolution_ = 40;
    std::size_t expected_count_;
    std::atomic<std::size_t> current_count_{0};
    std::string progress_info_;
    std::atomic<std::size_t> progress_pixel_{0};
    bool active_;
    std::atomic<bool> finalized_{false};
};

}  // namespace utility
}  // namespace open3d
