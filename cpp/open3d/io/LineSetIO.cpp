// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/LineSetIO.h"

#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

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
                "Read geometry::LineSet failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_lineset_read_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_read_function.end()) {
        utility::LogWarning(
                "Read geometry::LineSet failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, lineset, print_progress);
    utility::LogDebug("Read geometry::LineSet: {:d} vertices.",
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
                "Write geometry::LineSet failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_lineset_write_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_write_function.end()) {
        utility::LogWarning(
                "Write geometry::LineSet failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, lineset, write_ascii, compressed,
                                   print_progress);
    utility::LogDebug("Write geometry::LineSet: {:d} vertices.",
                      (int)lineset.points_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
