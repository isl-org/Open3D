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

#include "Open3D/IO/ClassIO/LineSetIO.h"

#include <unordered_map>

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::LineSet &, bool)>>
        file_extension_to_lineset_read_function{
                {"ply", ReadLineSetFromPLY},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::LineSet &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_lineset_write_function{
                {"ply", WriteLineSetToPLY},
        };
}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::LineSet> CreateLineSetFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto lineset = std::make_shared<geometry::LineSet>();
    ReadLineSet(filename, *lineset, format, print_progress);
    return lineset;
}

bool ReadLineSet(const std::string &filename,
                 geometry::LineSet &lineset,
                 const std::string &format,
                 bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::LineSet failed: unknown file extension.\n");
        return false;
    }
    auto map_itr = file_extension_to_lineset_read_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_read_function.end()) {
        utility::LogWarning(
                "Read geometry::LineSet failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, lineset, print_progress);
    utility::LogDebug("Read geometry::LineSet: {:d} vertices.\n",
                      (int)lineset.points_.size());
    return success;
}

bool WriteLineSet(const std::string &filename,
                  const geometry::LineSet &lineset,
                  bool write_ascii /* = false*/,
                  bool compressed /* = false*/,
                  bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::LineSet failed: unknown file extension.\n");
        return false;
    }
    auto map_itr = file_extension_to_lineset_write_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_write_function.end()) {
        utility::LogWarning(
                "Write geometry::LineSet failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, lineset, write_ascii, compressed,
                                   print_progress);
    utility::LogDebug("Write geometry::LineSet: {:d} vertices.\n",
                      (int)lineset.points_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
