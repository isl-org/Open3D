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

#include <fmt/format.h>

#include <memory>
#include <string>

#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

CompilerInfo::CompilerInfo() {}

CompilerInfo& CompilerInfo::GetInstance() {
    static CompilerInfo instance;
    return instance;
}

std::string CompilerInfo::CxxCompilerName() const {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

std::string CompilerInfo::CxxCompilerVersion() const {
#if defined(__clang__)
    return fmt::format("{}.{}.{}", __clang_major__, __clang_minor__,
                       __clang_patchlevel__);
#elif defined(__GNUC__)
    return fmt::format("{}.{}.{}", __GNUC__, __GNUC_MINOR__,
                       __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    // Support for VS 2010 or later.
    // https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros
    std::string vs_version;
    if (_MSC_VER >= 1930) {
        vs_version = "2022+";
    } else if (_MSC_VER >= 1920) {
        vs_version = "2019";
    } else if (_MSC_VER >= 1910) {
        vs_version = "2017";
    } else if (_MSC_VER == 1900) {
        vs_version = "2015";
    } else if (_MSC_VER == 1800) {
        vs_version = "2013";
    } else if (_MSC_VER == 1700) {
        vs_version = "2012";
    } else if (_MSC_VER == 1600) {
        vs_version = "2010";
    } else {
        vs_version = "unknown";
    }
    return fmt::format("{} (vs{})", _MSC_VER, vs_version);
#else
    return "unknown";
#endif
}

#ifdef BUILD_CUDA_MODULE
// See CompilerInfo.cu
#else
std::string CompilerInfo::CUDACompilerVersion() const { return "disabled"; }
#endif

std::string CompilerInfo::CxxStandard() const {
    if (__cplusplus == 202002L) {
        return "20";
    } else if (__cplusplus == 201703L) {
        return "17";
    } else if (__cplusplus == 201402L) {
        return "14";
    } else if (__cplusplus == 201103L) {
        return "11";
    } else if (__cplusplus == 199711L) {
        return "98";
    } else {
        return "unknown";
    }
}

void CompilerInfo::Print() const {
    utility::LogInfo("CompilerInfo: C++{}, {} {}, nvcc {}.", CxxStandard(),
                     CxxCompilerName(), CxxCompilerVersion(),
                     CUDACompilerVersion());
}

}  // namespace utility
}  // namespace open3d
