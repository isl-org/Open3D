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

#include <fmt/chrono.h>
#include <json/json.h>

#include <atomic>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "open3d/Open3D.h"

namespace open3d {
namespace apps {
namespace offline_reconstruction {

inline std::string PadZeroToNumber(int num, int size) {
    return fmt::format("{0:0{1}}", num, size);
}

inline std::string DurationToHMS(const double& ms) {
    std::chrono::system_clock::time_point t{
            std::chrono::milliseconds{size_t(ms)}};
    return fmt::format("{:%T}", t.time_since_epoch());
}

inline std::string FloatToString(float f, int precision = 3) {
    return fmt::format("{0:.{1}f}", f, precision);
}

bool CheckFolderStructure(const std::string& path_dataset) {
    if (utility::filesystem::FileExists(path_dataset) &&
        utility::filesystem::GetFileExtensionInLowerCase(path_dataset) ==
                "bag") {
        return true;
    }
    const std::string path_color = utility::filesystem::AddIfExist(
            path_dataset, {"color", "rgb", "image"});
    if (path_color == path_dataset) {
        utility::LogError("Can not find color folder in {}", path_dataset);
    }

    const std::string path_depth =
            utility::filesystem::JoinPath(path_dataset, "depth");
    if (!utility::filesystem::DirectoryExists(path_color) ||
        !utility::filesystem::DirectoryExists(path_depth)) {
        utility::LogWarning("Folder structure of {} is not correct",
                            path_dataset);
        return false;
    }
    return true;
}

void MakeCleanFolder(const std::string& path) {
    if (utility::filesystem::DirectoryExists(path)) {
        utility::filesystem::DeleteDirectory(path);
    }
    utility::filesystem::MakeDirectory(path);
}

bool ReadJsonFromFile(const std::string& path, Json::Value& json) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        utility::LogWarning("Failed to open {}", path);
        return false;
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::String errs;
    bool is_parse_successful = parseFromStream(builder, ifs, &json, &errs);
    if (!is_parse_successful) {
        utility::LogWarning("Read JSON failed: {}.", errs);
        return false;
    }
    return true;
}

std::tuple<std::string, std::string> GetRGBDFolders(
        const std::string& path_dataset) {
    return std::make_tuple(
            utility::filesystem::AddIfExist(path_dataset,
                                            {"image/", "rgb/", "color/"}),
            utility::filesystem::JoinPath(path_dataset, "depth/"));
}

std::tuple<std::vector<std::string>, std::vector<std::string>> ReadRGBDFiles(
        const std::string& path) {
    std::string path_rgb, path_depth;
    std::tie(path_rgb, path_depth) = GetRGBDFolders(path);

    std::vector<std::string> color_files, depth_files;
    utility::filesystem::ListFilesInDirectoryWithExtension(path_rgb, "png",
                                                           color_files);
    if (color_files.empty()) {
        utility::filesystem::ListFilesInDirectoryWithExtension(path_rgb, "jpg",
                                                               color_files);
    }
    utility::filesystem::ListFilesInDirectoryWithExtension(path_depth, "png",
                                                           depth_files);

    if (color_files.size() != depth_files.size()) {
        utility::LogError(
                "Number of color {} and depth {} images are not equal.",
                color_files.size(), depth_files.size());
    }
    return std::make_tuple(color_files, depth_files);
}

std::vector<std::string> ReadPlyFiles(const std::string& path) {
    std::vector<std::string> ply_files;
    utility::filesystem::ListFilesInDirectoryWithExtension(path, "ply",
                                                           ply_files);
    return ply_files;
}

std::tuple<std::string, std::string, float> ExtractRGBDFrames(
        const std::string& rgbd_video_file) {
    const std::string frames_folder =
            utility::filesystem::GetFileParentDirectory(rgbd_video_file);
    const std::string path_intrinsic = frames_folder + "intrinsic.json";
    if (!utility::filesystem::FileExists(path_intrinsic)) {
        utility::LogError("Intrinsic file not found: {}", path_intrinsic);
    } else {
        auto rgbd_video = t::io::RGBDVideoReader::Create(rgbd_video_file);
        rgbd_video->SaveFrames(frames_folder);
    }

    Json::Value intrinsic = utility::StringToJson(path_intrinsic);
    return std::make_tuple(frames_folder, path_intrinsic,
                           intrinsic["depth_scale"].asFloat());
}

}  // namespace offline_reconstruction
}  // namespace apps
}  // namespace open3d
