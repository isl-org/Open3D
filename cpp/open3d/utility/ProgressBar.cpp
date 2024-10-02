// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ProgressBar.h"

#include <fmt/printf.h>

#include "open3d/utility/Logging.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <tbb/tbb.h>

namespace open3d {
namespace utility {

ProgressBar::ProgressBar(std::size_t expected_count,
                         std::string progress_info,
                         bool active) {
    Reset(expected_count, std::move(progress_info), active);
}

void ProgressBar::Reset(std::size_t expected_count,
                        std::string progress_info,
                        bool active) {
    expected_count_ = expected_count;
    progress_info_ = std::move(progress_info);
    progress_pixel_ = 0;
    active_ = active;
    // Essentially set current count to zero but
    // goes through an overridden increment operator
    current_count_ = std::numeric_limits<std::size_t>::max();
    operator++();
}

ProgressBar& ProgressBar::operator+=(std::size_t n) {
    SetCurrentCount(current_count_ + 1);
    return *this;
}

void ProgressBar::SetCurrentCount(std::size_t n) {
    current_count_ = n;
    UpdateBar();
}

void ProgressBar::UpdateBar() {
    if (!active_) {
        return;
    }
    if (current_count_ >= expected_count_) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else {
        std::size_t new_progress_pixel =
                int(current_count_ * resolution_ / expected_count_);
        if (new_progress_pixel > progress_pixel_) {
            progress_pixel_ = new_progress_pixel;
            int percent = int(current_count_ * 100 / expected_count_);
            fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                       std::string(progress_pixel_, '='),
                       std::string(resolution_ - 1 - progress_pixel_, ' '),
                       percent);
            fflush(stdout);
        }
    }
}

OMPProgressBar::OMPProgressBar(std::size_t expected_count,
                               std::string progress_info,
                               bool active)
    : ProgressBar(expected_count, std::move(progress_info), active) {}

ProgressBar& OMPProgressBar::operator+=(std::size_t n) {
    // Ref: https://stackoverflow.com/a/44555438
#ifdef _OPENMP
    int number_of_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    if (number_of_threads > 1) {
        if (thread_id == 0) {
            SetCurrentCount(current_count_ + number_of_threads * n);
        }
    } else {
        SetCurrentCount(current_count_ + n);
    }
#else
    SetCurrentCount(current_count_ + n);
#endif
    return *this;
}

TBBProgressBar::TBBProgressBar(std::size_t expected_count,
                               std::string progress_info,
                               bool active) {
    Reset(expected_count, std::move(progress_info), active);
}

void TBBProgressBar::Reset(std::size_t expected_count,
                           std::string progress_info,
                           bool active) noexcept(false) {
    if (expected_count & flag_bit_mask) {
        utility::LogError("Expected count out of range [0, 2^31)");
    }
    expected_count_ = expected_count;
    progress_info_ = std::move(progress_info);
    progress_pixel_ = 0;
    active_ = active;
    current_count_ = 0;
    UpdateBar();
}

TBBProgressBar& TBBProgressBar::operator++() {
    ++current_count_;
    UpdateBar();
    return *this;
}

TBBProgressBar& TBBProgressBar::operator+=(std::size_t n) {
    current_count_ += n;
    UpdateBar();
    return *this;
}

void TBBProgressBar::SetCurrentCount(std::size_t n) {
    current_count_ = n;
    UpdateBar();
}

void TBBProgressBar::UpdateBar() {
    if (!active_ || current_count_ & flag_bit_mask) {
        return;
    }
    // Check if the current count equals the expected count
    // If so set the flag bit and print 100%
    // tmp is created so that compare_exchange doesn't modify expected_count
    if (std::size_t tmp = expected_count_;
        current_count_.compare_exchange_strong(
                tmp, expected_count_ | flag_bit_mask)) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else if (tbb::this_task_arena::current_thread_index() == 0) {
        std::size_t new_progress_pixel =
                current_count_ * resolution_ / expected_count_;
        if (new_progress_pixel > progress_pixel_) {
            progress_pixel_ = new_progress_pixel;
            int percent = int(current_count_ * 100 / expected_count_);
            fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                       std::string(progress_pixel_, '='),
                       std::string(resolution_ - 1 - progress_pixel_, ' '),
                       percent);
            fflush(stdout);
        }
    }
}

}  // namespace utility
}  // namespace open3d
