// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ProgressBar.h"

#include <fmt/printf.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

ProgressBar::ProgressBar(std::size_t expected_count,
                         std::string progress_info,
                         bool active) {
    Reset(expected_count, std::move(progress_info), active);
}
ProgressBar::ProgressBar(const ProgressBar& other)
    : expected_count_(other.expected_count_),
      current_count_(other.current_count_.load(std::memory_order_relaxed)),
      progress_info_(other.progress_info_),
      progress_pixel_(other.progress_pixel_.load(std::memory_order_relaxed)),
      active_(other.active_),
      finalized_(other.finalized_.load(std::memory_order_relaxed)) {}

ProgressBar& ProgressBar::operator=(const ProgressBar& other) {
    if (this != &other) {
        expected_count_ = other.expected_count_;
        current_count_.store(
                other.current_count_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        progress_info_ = other.progress_info_;
        progress_pixel_.store(
                other.progress_pixel_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        active_ = other.active_;
        finalized_.store(other.finalized_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }
    return *this;
}

void ProgressBar::Reset(std::size_t expected_count,
                        std::string progress_info,
                        bool active) {
    expected_count_ = expected_count;
    progress_info_ = std::move(progress_info);
    progress_pixel_ = 0;
    active_ = active;
    current_count_.store(0, std::memory_order_relaxed);
    progress_pixel_.store(0, std::memory_order_relaxed);
    finalized_.store(false, std::memory_order_relaxed);
}

ProgressBar& ProgressBar::operator+=(std::size_t n) {
    current_count_.fetch_add(n, std::memory_order_relaxed);
    UpdateBar();
    return *this;
}

void ProgressBar::SetCurrentCount(std::size_t n) {
    current_count_.store(n, std::memory_order_relaxed);
    UpdateBar();
}

void ProgressBar::UpdateBar() {
    if (!active_) {
        return;
    }
    const std::size_t current_count =
            current_count_.load(std::memory_order_relaxed);
    if (current_count >= expected_count_) {
        if (!finalized_.exchange(true, std::memory_order_relaxed)) {
            fmt::print("{}[{}] 100%\n", progress_info_,
                       std::string(resolution_, '='));
        }
        return;
    }
    const std::size_t new_progress_pixel =
            current_count * resolution_ / expected_count_;
    std::size_t previous_pixel =
            progress_pixel_.load(std::memory_order_relaxed);
    while (new_progress_pixel > previous_pixel &&
           !progress_pixel_.compare_exchange_weak(
                   previous_pixel, new_progress_pixel,
                   std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
    if (new_progress_pixel > previous_pixel) {
        const int percent = int(current_count * 100 / expected_count_);
        fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                   std::string(new_progress_pixel, '='),
                   std::string(resolution_ - 1 - new_progress_pixel, ' '),
                   percent);
        fflush(stdout);
    }
}

std::size_t ProgressBar::GetCurrentCount() const {
    return current_count_.load(std::memory_order_relaxed);
}

}  // namespace utility
}  // namespace open3d
