// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/ImageWarpingFieldIO.h"

#include <unordered_map>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

bool ReadImageWarpingFieldFromJSON(
        const std::string &filename,
        pipelines::color_map::ImageWarpingField &warping_field) {
    return ReadIJsonConvertible(filename, warping_field);
}

bool WriteImageWarpingFieldToJSON(
        const std::string &filename,
        const pipelines::color_map::ImageWarpingField &warping_field) {
    return WriteIJsonConvertibleToJSON(filename, warping_field);
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           pipelines::color_map::ImageWarpingField &)>>
        file_extension_to_warping_field_read_function{
                {"json", ReadImageWarpingFieldFromJSON},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const pipelines::color_map::ImageWarpingField &)>>
        file_extension_to_warping_field_write_function{
                {"json", WriteImageWarpingFieldToJSON},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<pipelines::color_map::ImageWarpingField>
CreateImageWarpingFieldFromFile(const std::string &filename) {
    auto warping_field =
            std::make_shared<pipelines::color_map::ImageWarpingField>();
    ReadImageWarpingField(filename, *warping_field);
    return warping_field;
}

bool ReadImageWarpingField(
        const std::string &filename,
        pipelines::color_map::ImageWarpingField &warping_field) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_warping_field_read_function.find(filename_ext);
    if (map_itr == file_extension_to_warping_field_read_function.end()) {
        utility::LogWarning(
                "Read pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, warping_field);
}

bool WriteImageWarpingField(
        const std::string &filename,
        const pipelines::color_map::ImageWarpingField &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_warping_field_write_function.find(filename_ext);
    if (map_itr == file_extension_to_warping_field_write_function.end()) {
        utility::LogWarning(
                "Write pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

}  // namespace io
}  // namespace open3d
