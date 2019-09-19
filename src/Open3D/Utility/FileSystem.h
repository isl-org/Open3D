// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include <vector>

namespace open3d {
namespace utility {
namespace filesystem {

std::string GetFileExtensionInLowerCase(const std::string &filename);

std::string GetFileNameWithoutExtension(const std::string &filename);

std::string GetFileNameWithoutDirectory(const std::string &filename);

std::string GetFileParentDirectory(const std::string &filename);

std::string GetRegularizedDirectoryName(const std::string &directory);

std::string GetWorkingDirectory();

bool ChangeWorkingDirectory(const std::string &directory);

bool DirectoryExists(const std::string &directory);

bool MakeDirectory(const std::string &directory);

bool MakeDirectoryHierarchy(const std::string &directory);

bool DeleteDirectory(const std::string &directory);

bool FileExists(const std::string &filename);

bool RemoveFile(const std::string &filename);

bool ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames);

bool ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames);

// wrapper for fopen that enables unicode paths on Windows
FILE *FOpen(const std::string &filename, const std::string &mode);

}  // namespace filesystem
}  // namespace utility
}  // namespace open3d
