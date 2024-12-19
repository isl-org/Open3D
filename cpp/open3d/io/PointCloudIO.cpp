// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/PointCloudIO.h"

#include <iostream>
#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           geometry::PointCloud &,
                           const ReadPointCloudOption &)>>
        file_extension_to_pointcloud_read_function{
                {"xyz", ReadPointCloudFromXYZ},
                {"xyzn", ReadPointCloudFromXYZN},
                {"xyzrgb", ReadPointCloudFromXYZRGB},
                {"ply", ReadPointCloudFromPLY},
                {"pcd", ReadPointCloudFromPCD},
                {"pts", ReadPointCloudFromPTS},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const unsigned char *,
                           const size_t,
                           geometry::PointCloud &,
                           const ReadPointCloudOption &)>>
        in_memory_to_pointcloud_read_function{
                {"mem::xyz", ReadPointCloudInMemoryFromXYZ},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::PointCloud &,
                           const WritePointCloudOption &)>>
        file_extension_to_pointcloud_write_function{
                {"xyz", WritePointCloudToXYZ},
                {"xyzn", WritePointCloudToXYZN},
                {"xyzrgb", WritePointCloudToXYZRGB},
                {"ply", WritePointCloudToPLY},
                {"pcd", WritePointCloudToPCD},
                {"pts", WritePointCloudToPTS},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(unsigned char *&,
                           size_t &,
                           const geometry::PointCloud &,
                           const WritePointCloudOption &)>>
        in_memory_to_pointcloud_write_function{
                {"mem::xyz", WritePointCloudInMemoryToXYZ},
        };

std::shared_ptr<geometry::PointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud, {format, true, true, print_progress});
    return pointcloud;
}

std::shared_ptr<geometry::PointCloud> CreatePointCloudFromMemory(
        const unsigned char *buffer,
        const size_t length,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(buffer, length, *pointcloud,
                   {format, true, true, print_progress});
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    utility::LogDebug("Format {} File {}", params.format, filename);

    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown file extension for "
                "{} (format: {}).",
                filename, params.format);
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, params);
    utility::LogDebug("Read geometry::PointCloud: {} vertices.",
                      pointcloud.points_.size());
    if (params.remove_nan_points || params.remove_infinite_points) {
        pointcloud.RemoveNonFinitePoints(params.remove_nan_points,
                                         params.remove_infinite_points);
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

    ReadPointCloudOption p;
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
bool ReadPointCloud(const unsigned char *buffer,
                    const size_t length,
                    geometry::PointCloud &pointcloud,
                    const ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }

    utility::LogDebug("Format {}", params.format);

    auto map_itr = in_memory_to_pointcloud_read_function.find(format);
    if (map_itr == in_memory_to_pointcloud_read_function.end()) {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }
    bool success = map_itr->second(buffer, length, pointcloud, params);
    utility::LogDebug("Read geometry::PointCloud: {} vertices.",
                      pointcloud.points_.size());
    if (params.remove_nan_points || params.remove_infinite_points) {
        pointcloud.RemoveNonFinitePoints(params.remove_nan_points,
                                         params.remove_infinite_points);
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     const WritePointCloudOption &params) {
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        utility::LogWarning(
                "Write geometry::PointCloud failed: unknown file extension {} "
                "for file {}.",
                format, filename);
        return false;
    }

    bool success = map_itr->second(filename, pointcloud, params);
    utility::LogDebug("Write geometry::PointCloud: {} vertices.",
                      pointcloud.points_.size());
    return success;
}
bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     const std::string &file_format /* = "auto"*/,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    WritePointCloudOption p;
    p.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = WritePointCloudOption::Compressed(compressed);
    std::string format = file_format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return WritePointCloud(filename, pointcloud, p);
}

bool WritePointCloud(unsigned char *&buffer,
                     size_t &length,
                     const geometry::PointCloud &pointcloud,
                     const WritePointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        utility::LogWarning(
                "Write geometry::PointCloud failed: unknown format for "
                "(format: {}).",
                params.format);
        return false;
    }
    auto map_itr = in_memory_to_pointcloud_write_function.find(format);
    if (map_itr == in_memory_to_pointcloud_write_function.end()) {
        utility::LogWarning(
                "Write geometry::PointCloud failed: unknown format {}.",
                format);
        return false;
    }

    bool success = map_itr->second(buffer, length, pointcloud, params);
    utility::LogDebug("Write geometry::PointCloud: {} vertices.",
                      pointcloud.points_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
