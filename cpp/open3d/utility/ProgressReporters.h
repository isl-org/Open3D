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

#include <functional>
#include <string>

#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace utility {

/// Progress reporting through update_progress(double percent) function.
/// If you have a set number of items to process (or bytes to load),
/// CountingProgressReporter will convert that to percentages (you still have to
/// specify how many items you have, of course)
class CountingProgressReporter {
public:
    CountingProgressReporter(std::function<bool(double)> f) {
        update_progress_ = f;
    }
    void SetTotal(int64_t total) { total_ = total; }
    bool Update(int64_t count) {
        if (!update_progress_) return true;
        last_count_ = count;
        double percent = 0;
        if (total_ > 0) {
            if (count < total_) {
                percent = count * 100.0 / total_;
            } else {
                percent = 100.0;
            }
        }
        return CallUpdate(percent);
    }
    void Finish() { CallUpdate(100); }
    // for compatibility with ProgressBar
    void operator++() { Update(last_count_ + 1); }

private:
    bool CallUpdate(double percent) {
        if (update_progress_) {
            return update_progress_(percent);
        }
        return true;
    }
    std::function<bool(double)> update_progress_;
    int64_t total_ = -1;
    int64_t last_count_ = -1;
};

/// update_progress(double percent) functor for ProgressBar
struct ConsoleProgressUpdater {
    ConsoleProgressUpdater(const std::string &progress_info,
                           bool active = false)
        : progress_bar_(100, progress_info, active) {}
    bool operator()(double pct) {
        while (last_pct_ < pct) {
            ++last_pct_;
            ++progress_bar_;
        }
        return true;
    }

private:
    utility::ProgressBar progress_bar_;
    int last_pct_ = 0;
};

}  // namespace utility
}  // namespace open3d
