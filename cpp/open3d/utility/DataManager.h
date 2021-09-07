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

/// \file DataManager.h
///
/// TEST_DATA_DIR should be defined as a compile flag. The relative paths are
/// computed from TEST_DATA_DIR. The code that uses TEST_DATA_DIR should be
/// header-only, that is, we should not bake in a TEST_DATA_DIR value in the
/// Open3D main library.

#pragma once

// Avoid editor warnings.
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR
#endif

#include <string>

namespace open3d {
namespace utility {

/// Computes the full path of a file/directory inside the Open3D common data
/// root. If \p relative_path is specified, the full path is computed by
/// appending the \p relative_path to the common data root; otherwise, the
/// common data root is returned.
///
/// \param relative_path Relative path to Open3D common data root.
inline std::string GetDataPathCommon(const std::string& relative_path = "") {
    if (relative_path.empty()) {
        return std::string(TEST_DATA_DIR);
    } else {
        return std::string(TEST_DATA_DIR) + "/" + relative_path;
    }
}

/// Computes the full path of a file/directory inside the Open3D download data
/// root. If \p relative_path is specified, the full path is computed by
/// appending the \p relative_path to the download data root; otherwise, the
/// download data root is returned.
///
/// \param relative_path Relative path to Open3D download data root.
inline std::string GetDataPathDownload(const std::string& relative_path = "") {
    if (relative_path.empty()) {
        return std::string(TEST_DATA_DIR) + "/open3d_downloads";
    } else {
        return std::string(TEST_DATA_DIR) + "/open3d_downloads/" +
               relative_path;
    }
}

}  // namespace utility
}  // namespace open3d
