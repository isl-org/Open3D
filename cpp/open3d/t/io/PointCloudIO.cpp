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

#include "open3d/t/io/PointCloudIO.h"

#include <iostream>
#include <unordered_map>

#include "open3d/io/PointCloudIO.h"
#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           geometry::PointCloud &,
                           const open3d::io::ReadPointCloudOption &)>>
        file_extension_to_pointcloud_read_function{
                {"npz", ReadPointCloudFromNPZ},
                {"xyz", ReadPointCloudFromTXT},
                {"xyzi", ReadPointCloudFromTXT},
                {"xyzn", ReadPointCloudFromTXT},
                {"xyzrgb", ReadPointCloudFromTXT},
                {"pcd", ReadPointCloudFromPCD},
                {"ply", ReadPointCloudFromPLY},
                {"pts", ReadPointCloudFromPTS},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::PointCloud &,
                           const open3d::io::WritePointCloudOption &)>>
        file_extension_to_pointcloud_write_function{
                {"npz", WritePointCloudToNPZ},
                {"xyz", WritePointCloudToTXT},
                {"xyzi", WritePointCloudToTXT},
                {"xyzn", WritePointCloudToTXT},
                {"xyzrgb", WritePointCloudToTXT},
                {"pcd", WritePointCloudToPCD},
                {"ply", WritePointCloudToPLY},
                {"pts", WritePointCloudToPTS},
        };

std::shared_ptr<geometry::PointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud,
                   {format, false, false, print_progress});
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const open3d::io::ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    utility::LogDebug("Format {} File {}", params.format, filename);

    bool success = false;
    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown file extension for "
                "{} (format: {}).",
                filename, params.format);
        return false;
    } else {
        success = map_itr->second(filename, pointcloud, params);
        if (params.remove_nan_points || params.remove_infinite_points) {
            utility::LogError(
                    "remove_nan_points and remove_infinite_points options are "
                    "unimplemented.");
            return false;
        }
    }

    utility::LogDebug(
            "Read t::geometry::PointCloud with following attributes: ");
    for (auto &kv : pointcloud.GetPointAttr()) {
        utility::LogDebug(" {} [shape: {}, stride: {}, {}]", kv.first,
                          kv.second.GetShape().ToString(),
                          kv.second.GetStrides().ToString(),
                          kv.second.GetDtype().ToString());
    }

    return success;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const std::string &file_format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string format = file_format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    open3d::io::ReadPointCloudOption p;
    p.format = format;
    p.remove_nan_points = remove_nan_points;
    p.remove_infinite_points = remove_infinite_points;
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Reading ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return ReadPointCloud(filename, pointcloud, p);
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     const open3d::io::WritePointCloudOption &params) {
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        return open3d::io::WritePointCloud(filename, pointcloud.ToLegacy(),
                                           params);
    }

    bool success = map_itr->second(
            filename, pointcloud.To(core::Device("CPU:0")), params);
    if (!pointcloud.IsEmpty()) {
        utility::LogDebug("Write geometry::PointCloud: {:d} vertices.",
                          (int)pointcloud.GetPointPositions().GetLength());
    } else {
        utility::LogDebug("Write geometry::PointCloud: 0 vertices.");
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    open3d::io::WritePointCloudOption p;
    p.write_ascii = open3d::io::WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = open3d::io::WritePointCloudOption::Compressed(compressed);
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return WritePointCloud(filename, pointcloud, p);
}

bool ReadPointCloudFromNPZ(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    // Required checks are performed in the pointcloud constructor itself.
    pointcloud = geometry::PointCloud(ReadNpz(filename));
    return true;
}

bool WritePointCloudToNPZ(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    if (bool(params.write_ascii)) {
        utility::LogError("PointCloud can't be saved in ASCII format as .npz.");
    }
    // TODO: When open3d NPZ io supports compression in future, update this.
    if (bool(params.compressed)) {
        utility::LogError(
                "PointCloud can't be saved in compressed format as .npz.");
    }

    WriteNpz(filename, pointcloud.GetPointAttr());
    utility::LogDebug("Saved pointcloud has the following attributes:");
    for (auto &kv : pointcloud.GetPointAttr()) {
        utility::LogDebug(" {} [shape: {}, stride: {}, {}]", kv.first,
                          kv.second.GetShape().ToString(),
                          kv.second.GetStrides().ToString(),
                          kv.second.GetDtype().ToString());
    }

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
