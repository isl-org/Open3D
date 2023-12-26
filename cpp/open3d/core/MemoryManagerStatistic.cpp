// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/MemoryManagerStatistic.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

MemoryManagerStatistic& MemoryManagerStatistic::GetInstance() {
    // Ensure the static Logger instance is instantiated before the
    // MemoryManagerStatistic instance.
    // Since destruction of static instances happens in reverse order,
    // this guarantees that the Logger can be used at any point in time.
    utility::Logger::GetInstance();

    static MemoryManagerStatistic instance;
    return instance;
}

MemoryManagerStatistic::~MemoryManagerStatistic() {
    if (print_at_program_end_) {
        // Always use the default print function (print to the console).
        // Custom print functions like py::print may not work reliably
        // at this point in time.
        utility::Logger::GetInstance().ResetPrintFunction();
        Print();

        // Indicate failure if possible leaks have been detected.
        // This is useful to automatically let unit tests fail.
        if (HasLeaks()) {
            std::exit(EXIT_FAILURE);
        }
    }
}

void MemoryManagerStatistic::SetPrintLevel(PrintLevel level) { level_ = level; }

void MemoryManagerStatistic::SetPrintAtProgramEnd(bool print) {
    print_at_program_end_ = print;
}

void MemoryManagerStatistic::SetPrintAtMallocFree(bool print) {
    print_at_malloc_free_ = print;
}

void MemoryManagerStatistic::Print() const {
    if (level_ == PrintLevel::None) {
        return;
    }

    if (level_ == PrintLevel::Unbalanced && !HasLeaks()) {
        return;
    }

    // Ensure all information gets printed.
    auto old_level = utility::GetVerbosityLevel();
    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    utility::LogInfo("Memory Statistics: (Device) (#Malloc) (#Free)");
    utility::LogInfo("---------------------------------------------");
    for (const auto& value_pair : statistics_) {
        // Simulate C++17 structured bindings for better readability.
        const auto& device = value_pair.first;
        const auto& statistics = value_pair.second;

        if (level_ == PrintLevel::Unbalanced && statistics.IsBalanced()) {
            continue;
        }

        if (!statistics.IsBalanced()) {
            int64_t count_leaking =
                    statistics.count_malloc_ - statistics.count_free_;

            size_t leaking_byte_size = std::accumulate(
                    statistics.active_allocations_.begin(),
                    statistics.active_allocations_.end(), 0,
                    [](size_t count, auto ptr_byte_size) -> size_t {
                        return count + ptr_byte_size.second;
                    });

            utility::LogWarning("{}: {} {} --> {} with {} total bytes",
                                device.ToString(), statistics.count_malloc_,
                                statistics.count_free_, count_leaking,
                                leaking_byte_size);

            for (const auto& leak : statistics.active_allocations_) {
                utility::LogWarning("    {} @ {} bytes", fmt::ptr(leak.first),
                                    leak.second);
            }
        } else {
            utility::LogInfo("{}: {} {}", device.ToString(),
                             statistics.count_malloc_, statistics.count_free_);
        }
    }
    utility::LogInfo("---------------------------------------------");

    // Restore old verbosity level.
    utility::SetVerbosityLevel(old_level);
}

bool MemoryManagerStatistic::HasLeaks() const {
    return std::any_of(statistics_.begin(), statistics_.end(),
                       [](const auto& value_pair) -> bool {
                           return !value_pair.second.IsBalanced();
                       });
}

void MemoryManagerStatistic::CountMalloc(void* ptr,
                                         size_t byte_size,
                                         const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);

    // Filter nullptr. Empty allocations are not tracked.
    if (ptr == nullptr && byte_size == 0) {
        return;
    }

    auto it = statistics_[device].active_allocations_.emplace(ptr, byte_size);
    if (it.second) {
        statistics_[device].count_malloc_++;
        if (print_at_malloc_free_) {
            utility::LogInfo("[Malloc] {}: {} @ {} bytes",
                             fmt::sprintf("%6s", device.ToString()),
                             fmt::ptr(ptr), byte_size);
        }
    } else {
        utility::LogError(
                "{} @ {} bytes on {} is still active and was not freed before",
                fmt::ptr(ptr), byte_size, device.ToString());
    }
}

void MemoryManagerStatistic::CountFree(void* ptr, const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);

    // Filter nullptr. Empty allocations are not tracked.
    if (ptr == nullptr) {
        return;
    }

    auto num_to_erase = statistics_[device].active_allocations_.count(ptr);
    if (num_to_erase == 1) {
        if (print_at_malloc_free_) {
            utility::LogInfo("[ Free ] {}: {} @ {} bytes",
                             fmt::sprintf("%6s", device.ToString()),
                             fmt::ptr(ptr),
                             statistics_[device].active_allocations_.at(ptr));
        }
        statistics_[device].active_allocations_.erase(ptr);
        statistics_[device].count_free_++;
    } else if (num_to_erase == 0) {
        // Either the statistics were reset before or the given pointer is
        // invalid. Do not increase any counts and ignore both cases.
    } else {
        // Should never reach here.
        utility::LogError(
                "Invalid number of erased allocations {} for {} on {}",
                num_to_erase, fmt::ptr(ptr), device.ToString());
    }
}

void MemoryManagerStatistic::Reset() {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_.clear();
}

bool MemoryManagerStatistic::MemoryStatistics::IsBalanced() const {
    return count_malloc_ == count_free_;
}

}  // namespace core
}  // namespace open3d
