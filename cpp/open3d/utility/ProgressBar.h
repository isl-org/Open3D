// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include <tbb/task_scheduler_observer.h>

namespace open3d {
namespace utility {

class ProgressBar {
public:
    ProgressBar(size_t expected_count,
                const std::string &progress_info,
                bool active = false);
    void Reset(size_t expected_count,
               const std::string &progress_info,
               bool active);
    virtual ProgressBar &operator++();
    void SetCurrentCount(size_t n);
    size_t GetCurrentCount() const;
    virtual ~ProgressBar() = default;

protected:
    const size_t resolution_ = 40;
    size_t expected_count_;
    size_t current_count_;
    std::string progress_info_;
    size_t progress_pixel_;
    bool active_;
};

class OMPProgressBar : public ProgressBar {
public:
    OMPProgressBar(size_t expected_count,
                   const std::string &progress_info,
                   bool active = false);
    ProgressBar &operator++() override;
};

class TBBProgressBar : public ProgressBar, tbb::task_scheduler_observer {
public:
    std::atomic<int> num_threads;
    TBBProgressBar(std::size_t expected_count,
                   const std::string &progress_info,
                   bool active = false);
    ProgressBar &operator++() override;

    void on_scheduler_entry(bool is_worker) override;
    void on_scheduler_exit(bool is_worker) override;
};

}  // namespace utility
}  // namespace open3d
