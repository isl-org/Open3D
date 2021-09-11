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

/// Set of known ISA targets.
enum class ISATarget {
    /* x86 */
    SSE2 = 100,
    SSE4 = 101,
    AVX = 102,
    AVX2 = 103,
    AVX512KNL = 104,
    AVX512SKX = 105,
    /* ARM */
    NEON = 200,
    /* GPU */
    GENX = 300,
    /* Special values */
    UNKNOWN = -1,
    /* Additional value for disabled support */
    DISABLED = -100
};

/// \brief ISA information.
///
/// This provides information about kernel code written in ISPC.
class ISAInfo {
public:
    static ISAInfo& GetInstance();

    ~ISAInfo() = default;
    ISAInfo(const ISAInfo&) = delete;
    void operator=(const ISAInfo&) = delete;

    /// Returns the dispatched ISA target that will be used in kernel code.
    ISATarget SelectedTarget() const;

    /// Prints ISAInfo in the console.
    void Print() const;

private:
    ISAInfo();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace utility
}  // namespace open3d
