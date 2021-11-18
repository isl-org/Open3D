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

#include <string>

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

}  // namespace utility
}  // namespace open3d
