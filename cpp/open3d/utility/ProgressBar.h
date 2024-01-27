// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/collaborative_call_once.h>

#include <atomic>
#include <string>

namespace open3d {
namespace utility {

class ProgressBar {
public:
    ProgressBar(size_t expected_count,
                std::string progress_info,
                bool active = false);
    void Reset(size_t expected_count,
               std::string progress_info,
               bool active);
    inline ProgressBar& operator++() { return *this += 1; };
    virtual ProgressBar& operator+=(std::size_t n);
    void SetCurrentCount(size_t n);
    void UpdateBar();
    inline std::size_t GetCurrentCount() const { return current_count_; }
    virtual ~ProgressBar() = default;

protected:
    static constexpr size_t resolution_ = 40;
    std::size_t expected_count_;
    std::size_t current_count_;
    std::string progress_info_;
    std::size_t progress_pixel_;
    bool active_;
};

class OMPProgressBar : public ProgressBar {
public:
    OMPProgressBar(std::size_t expected_count,
                   std::string progress_info,
                   bool active = false);
    ProgressBar& operator+=(std::size_t) override;
};

class TBBProgressBar {
public:
    TBBProgressBar(std::size_t expected_count,
                   std::string progress_info,
                   bool active = false);
    void Reset(std::size_t expected_count,
               std::string progress_info,
               bool active);
    TBBProgressBar& operator++();
    TBBProgressBar& operator+=(std::size_t n);
    void SetCurrentCount(std::size_t n);
    void UpdateBar();
    inline std::size_t GetCurrentCount() const {
        return current_count_ & ~flag_bit_mask;
    }


protected:
    static constexpr std::size_t flag_bit_mask = ~(~std::size_t{} >> 1);
    static constexpr std::size_t resolution_ = 40;
    std::atomic<std::size_t> current_count_;
    tbb::collaborative_once_flag finalized;
    std::size_t expected_count_;
    std::string progress_info_;
    mutable std::size_t progress_pixel_;
    bool active_;
};

}  // namespace utility
}  // namespace open3d
