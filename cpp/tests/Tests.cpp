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

// TEST_DATA_DIR defined in CMakeLists.txt
// Put it here to avoid editor warnings
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR
#endif

#include "tests/Tests.h"

namespace open3d {
namespace tests {

void NotImplemented() {
    std::cout << "\033[0;32m"
              << "[          ] "
              << "\033[0;0m";
    std::cout << "\033[0;31m"
              << "Not implemented."
              << "\033[0;0m" << std::endl;

    GTEST_NONFATAL_FAILURE_("Not implemented");
}

std::string GetDataPathCommon(const std::string& relative_path) {
    if (relative_path.empty()) {
        return std::string(TEST_DATA_DIR);
    } else {
        return std::string(TEST_DATA_DIR) + "/" + relative_path;
    }
}

std::string GetDataPathDownload(const std::string& relative_path) {
    if (relative_path.empty()) {
        return std::string(TEST_DATA_DIR) + "/open3d_downloads";
    } else {
        return std::string(TEST_DATA_DIR) + "/open3d_downloads/" +
               relative_path;
    }
}

}  // namespace tests
}  // namespace open3d
