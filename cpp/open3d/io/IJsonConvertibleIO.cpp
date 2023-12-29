// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/IJsonConvertibleIO.h"

#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, utility::IJsonConvertible &)>>
        file_extension_to_ijsonconvertible_read_function{
                {"json", ReadIJsonConvertibleFromJSON},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const utility::IJsonConvertible &)>>
        file_extension_to_ijsonconvertible_write_function{
                {"json", WriteIJsonConvertibleToJSON},
        };

}  // unnamed namespace

namespace io {

bool ReadIJsonConvertible(const std::string &filename,
                          utility::IJsonConvertible &object) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read utility::IJsonConvertible failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_ijsonconvertible_read_function.find(filename_ext);
    if (map_itr == file_extension_to_ijsonconvertible_read_function.end()) {
        utility::LogWarning(
                "Read utility::IJsonConvertible failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, object);
}

bool WriteIJsonConvertible(const std::string &filename,
                           const utility::IJsonConvertible &object) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write utility::IJsonConvertible failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr = file_extension_to_ijsonconvertible_write_function.find(
            filename_ext);
    if (map_itr == file_extension_to_ijsonconvertible_write_function.end()) {
        utility::LogWarning(
                "Write utility::IJsonConvertible failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, object);
}

}  // namespace io
}  // namespace open3d
