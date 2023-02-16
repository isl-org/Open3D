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
#include <string>

namespace open3d {
namespace utility {

/// \brief Compiler information.
class CompilerInfo {
    // This does not need to be a class. It is a class just for the sake of
    // consistency with CPUInfo.
public:
    static CompilerInfo& GetInstance();

    ~CompilerInfo() = default;
    CompilerInfo(const CompilerInfo&) = delete;
    void operator=(const CompilerInfo&) = delete;

    std::string CXXStandard() const;

    std::string CXXCompilerId() const;
    std::string CXXCompilerVersion() const;

    std::string CUDACompilerId() const;
    std::string CUDACompilerVersion() const;

    void Print() const;

private:
    CompilerInfo();
};

}  // namespace utility
}  // namespace open3d
