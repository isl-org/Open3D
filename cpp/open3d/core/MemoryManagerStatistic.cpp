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

#include "open3d/core/MemoryManagerStatistic.h"

#include <algorithm>
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
    }
}

void MemoryManagerStatistic::SetPrintLevel(PrintLevel level) { level_ = level; }

void MemoryManagerStatistic::SetPrintAtProgramEnd(bool print) {
    print_at_program_end_ = print;
}

void MemoryManagerStatistic::Print() const {
    if (level_ == PrintLevel::None) {
        return;
    }

    auto is_unbalanced = [](const auto& value_pair) -> bool {
        return value_pair.second.count_malloc_ != value_pair.second.count_free_;
    };

    if (level_ == PrintLevel::Unbalanced &&
        std::count_if(statistics_.begin(), statistics_.end(), is_unbalanced) ==
                0) {
        return;
    }

    utility::LogInfo("Memory Statistics: (Device) (#Malloc) (#Free)");
    utility::LogInfo("---------------------------------------------");
    for (const auto& value_pair : statistics_) {
        if (level_ == PrintLevel::Unbalanced && !is_unbalanced(value_pair)) {
            continue;
        }

        if (is_unbalanced(value_pair)) {
            size_t count_leaking = value_pair.second.count_malloc_ -
                                   value_pair.second.count_free_;

            size_t leaking_byte_size = std::accumulate(
                    value_pair.second.active_allocations_.begin(),
                    value_pair.second.active_allocations_.end(), 0,
                    [](size_t count, auto ptr_byte_size) -> size_t {
                        return count + ptr_byte_size.second;
                    });

            utility::LogWarning("{}: {} {} --> {} with {} bytes",
                                value_pair.first.ToString(),
                                value_pair.second.count_malloc_,
                                value_pair.second.count_free_, count_leaking,
                                leaking_byte_size);
        } else {
            utility::LogInfo("{}: {} {}", value_pair.first.ToString(),
                             value_pair.second.count_malloc_,
                             value_pair.second.count_free_);
        }
    }
    utility::LogInfo("---------------------------------------------");
}

void MemoryManagerStatistic::IncrementCountMalloc(void* ptr,
                                                  size_t byte_size,
                                                  const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[device].count_malloc_++;
    statistics_[device].active_allocations_.emplace(ptr, byte_size);
}

void MemoryManagerStatistic::IncrementCountFree(void* ptr,
                                                const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[device].count_free_++;
    statistics_[device].active_allocations_.erase(ptr);
}

}  // namespace core
}  // namespace open3d
