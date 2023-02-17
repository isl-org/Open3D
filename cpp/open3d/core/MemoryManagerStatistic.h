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

#pragma once

#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {

class MemoryManagerStatistic {
public:
    enum class PrintLevel {
        /// Statistics for all used devices are printed.
        All = 0,
        /// Only devices with unbalanced counts are printed.
        /// This is typically an indicator for memory leaks.
        Unbalanced = 1,
        /// No statistics are printed.
        None = 2,
    };

    static MemoryManagerStatistic& GetInstance();

    MemoryManagerStatistic(const MemoryManagerStatistic&) = delete;
    MemoryManagerStatistic& operator=(MemoryManagerStatistic&) = delete;

    ~MemoryManagerStatistic();

    /// Sets the level of provided information for printing.
    void SetPrintLevel(PrintLevel level);

    /// Enables or disables printing at the program end.
    /// Printing at the program end additionally overrides the exit code
    /// to EXIT_FAILURE in the presence of leaks.
    void SetPrintAtProgramEnd(bool print);

    /// Enables or disables printing at each malloc and free.
    void SetPrintAtMallocFree(bool print);

    /// Prints statistics for all recorded devices depending on the print level.
    void Print() const;

    /// Returns true if any recorded device has unbalanced counts, false
    /// otherwise.
    bool HasLeaks() const;

    /// Adds the given allocation to the statistics.
    void CountMalloc(void* ptr, size_t byte_size, const Device& device);

    /// Adds the given deallocations to the statistics.
    /// Counts for previously recorded allocations after a reset are ignored for
    /// consistency.
    void CountFree(void* ptr, const Device& device);

    /// Resets the statistics.
    void Reset();

private:
    MemoryManagerStatistic() = default;

    struct MemoryStatistics {
        bool IsBalanced() const;

        int64_t count_malloc_ = 0;
        int64_t count_free_ = 0;
        std::unordered_map<void*, size_t> active_allocations_;
    };

    /// Only print unbalanced statistics by default.
    PrintLevel level_ = PrintLevel::Unbalanced;

    /// Print at the end of the program, enabled by default. If leaks are
    /// detected, the exit code will be overridden with EXIT_FAILURE.
    bool print_at_program_end_ = true;

    /// Print at each malloc and free, disabled by default.
    bool print_at_malloc_free_ = false;

    std::mutex statistics_mutex_;
    std::map<Device, MemoryStatistics> statistics_;
};

}  // namespace core
}  // namespace open3d
