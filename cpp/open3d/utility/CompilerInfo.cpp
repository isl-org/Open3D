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

#include "open3d/utility/CompilerInfo.h"

#include <memory>
#include <string>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

CompilerInfo::CompilerInfo() {}

CompilerInfo& CompilerInfo::GetInstance() {
    static CompilerInfo instance;
    return instance;
}

std::string CompilerInfo::CXXStandard() const {
    return std::string(OPEN3D_CXX_STANDARD);
}

std::string CompilerInfo::CXXCompilerId() const {
    return std::string(OPEN3D_CXX_COMPILER_ID);
}

std::string CompilerInfo::CXXCompilerVersion() const {
    return std::string(OPEN3D_CXX_COMPILER_VERSION);
}

std::string CompilerInfo::CUDACompilerId() const {
    return std::string(OPEN3D_CUDA_COMPILER_ID);
}

std::string CompilerInfo::CUDACompilerVersion() const {
    return std::string(OPEN3D_CUDA_COMPILER_VERSION);
}

void CompilerInfo::Print() const {
#ifdef BUILD_CUDA_MODULE
    utility::LogInfo("CompilerInfo: C++ {}, {} {}, {} {}.", CXXStandard(),
                     CXXCompilerId(), CXXCompilerVersion(), CUDACompilerId(),
                     CUDACompilerVersion());
#else
    utility::LogInfo("CompilerInfo: C++ {}, {} {}, CUDA disabled.",
                     CXXStandard(), CXXCompilerId(), CXXCompilerVersion());
#endif
}

}  // namespace utility
}  // namespace open3d
