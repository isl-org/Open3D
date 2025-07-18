// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ProgressBar.h"

#include <fmt/printf.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace open3d {
namespace utility {

ProgressBar::ProgressBar(size_t expected_count,
                         const std::string &progress_info,
                         bool active) {
    Reset(expected_count, progress_info, active);
}

void ProgressBar::Reset(size_t expected_count,
                        const std::string &progress_info,
                        bool active) {
    expected_count_ = expected_count;
    current_count_ = static_cast<size_t>(-1);  // Guaranteed to wraparound
    progress_info_ = progress_info;
    progress_pixel_ = 0;
    active_ = active;
    operator++();
}

ProgressBar &ProgressBar::operator++() {
    SetCurrentCount(current_count_ + 1);
    return *this;
}

void ProgressBar::SetCurrentCount(size_t n) {
    current_count_ = n;
    if (!active_) {
        return;
    }
    if (current_count_ >= expected_count_) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else {
        size_t new_progress_pixel =
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

size_t ProgressBar::GetCurrentCount() const { return current_count_; }

OMPProgressBar::OMPProgressBar(size_t expected_count,
                               const std::string &progress_info,
                               bool active)
    : ProgressBar(expected_count, progress_info, active) {}

ProgressBar &OMPProgressBar::operator++() {
    // Ref: https://stackoverflow.com/a/44555438
#ifdef _OPENMP
    int number_of_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    if (number_of_threads > 1) {
        if (thread_id == 0) {
            SetCurrentCount(current_count_ + number_of_threads);
        }
    } else {
        SetCurrentCount(current_count_ + 1);
    }
#else
    SetCurrentCount(current_count_ + 1);
#endif
    return *this;
}

}  // namespace utility
}  // namespace open3d
