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

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

MemoryManagerStatistic& MemoryManagerStatistic::getInstance() {
    // Ensure the static Logger instance is instantiated before the
    // MemoryManagerStatistic instance.
    // Since destruction of static instances happens in reverse order,
    // this guarantees that the Logger can be used at any point in time.
    open3d::utility::Logger::GetInstance();

    static MemoryManagerStatistic instance;
    return instance;
}

void MemoryManagerStatistic::setPrintLevel(PrintLevel level) { level_ = level; }

void MemoryManagerStatistic::Print() const {
    if (level_ == PrintLevel::None) {
        return;
    }

    auto is_unbalanced = [](const auto& value_pair) {
        return value_pair.second.count_malloc_ != value_pair.second.count_free_;
    };

    if (level_ == PrintLevel::Unbalanced &&
        std::count_if(statistics_.begin(), statistics_.end(), is_unbalanced) ==
                0) {
        return;
    }

    open3d::utility::LogInfo("Memory Statistics: (Device) (#Malloc) (#Free)");
    open3d::utility::LogInfo("---------------------------------------------");
    for (const auto& value_pair : statistics_) {
        if (level_ == PrintLevel::Unbalanced && !is_unbalanced(value_pair)) {
            continue;
        }

        if (is_unbalanced(value_pair)) {
            open3d::utility::LogWarning("{}: {} {} --> {}",
                                        value_pair.first.ToString(),
                                        value_pair.second.count_malloc_,
                                        value_pair.second.count_free_,
                                        value_pair.second.count_malloc_ -
                                                value_pair.second.count_free_);
        } else {
            open3d::utility::LogInfo("{}: {} {}", value_pair.first.ToString(),
                                     value_pair.second.count_malloc_,
                                     value_pair.second.count_free_);
        }
    }
    open3d::utility::LogInfo("---------------------------------------------");
}

void MemoryManagerStatistic::IncrementCountMalloc(const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[device].count_malloc_++;
}

void MemoryManagerStatistic::IncrementCountFree(const Device& device) {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[device].count_free_++;
}

}  // namespace core
}  // namespace open3d
