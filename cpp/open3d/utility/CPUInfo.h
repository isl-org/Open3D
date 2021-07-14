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

#include <memory>

namespace open3d {
namespace utility {

/// \brief CPU information.
class CPUInfo {
public:
    static CPUInfo& GetInstance();

    ~CPUInfo() = default;
    CPUInfo(const CPUInfo&) = delete;
    void operator=(const CPUInfo&) = delete;

    /// Returns the number of physical CPU cores.
    /// This is similar to boost::thread::physical_concurrency().
    int NumCores() const;

    /// Returns the number of logical CPU cores.
    /// This returns the same result as std::thread::hardware_concurrency() or
    /// boost::thread::hardware_concurrency().
    int NumThreads() const;

    /// Prints CPUInfo in the console.
    void Print() const;

private:
    CPUInfo();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace utility
}  // namespace open3d
