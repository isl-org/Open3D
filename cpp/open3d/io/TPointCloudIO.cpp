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

#include "open3d/io/TPointCloudIO.h"

#include <iostream>
#include <unordered_map>

#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           t::geometry::PointCloud &,
                           const ReadPointCloudOption &)>>
        file_extension_to_pointcloud_read_function{
                {"xyzi", ReadPointCloudFromXYZI},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const t::geometry::PointCloud &,
                           const WritePointCloudOption &)>>
        file_extension_to_pointcloud_write_function{
                {"xyzi", WritePointCloudToXYZI},
        };

std::shared_ptr<t::geometry::PointCloud> CreatetPointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<t::geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud, {format, true, true, print_progress});
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    t::geometry::PointCloud &pointcloud,
                    const ReadPointCloudOption &params) {
    std::string format = params.format;
    if (format == "auto") {
        format = utility::filesystem::GetFileExtensionInLowerCase(filename);
    }

    utility::LogDebug("Format {} File {}", params.format, filename);

    auto map_itr = file_extension_to_pointcloud_read_function.find(format);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::LogWarning(
                "Read t::geometry::PointCloud failed: unknown file extension "
                "for "
                "{} (format: {}).",
                filename, params.format);
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, params);
    utility::LogDebug("Read t::geometry::PointCloud: {:d} vertices.",
                      (int)pointcloud.GetPoints().GetSize());
    if (params.remove_nan_points || params.remove_infinite_points) {
        utility::LogError("Unimplemented");
        return false;
    }
    return success;
}

bool ReadPointCloud(const std::string &filename,
                    t::geometry::PointCloud &pointcloud,
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

bool WritePointCloud(const std::string &filename,
                     const t::geometry::PointCloud &pointcloud,
                     const WritePointCloudOption &params) {
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    auto map_itr = file_extension_to_pointcloud_write_function.find(format);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        utility::LogWarning(
                "Write t::geometry::PointCloud failed: unknown file extension "
                "{} "
                "for file {}.",
                format, filename);
        return false;
    }

    bool success = map_itr->second(filename, pointcloud, params);
    utility::LogDebug("Write t::geometry::PointCloud: {:d} vertices.",
                      (int)pointcloud.GetPoints().GetSize());
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const t::geometry::PointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    WritePointCloudOption p;
    p.write_ascii = WritePointCloudOption::IsAscii(write_ascii);
    p.compressed = WritePointCloudOption::Compressed(compressed);
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    utility::ConsoleProgressUpdater progress_updater(
            std::string("Writing ") + utility::ToUpper(format) +
                    " file: " + filename,
            print_progress);
    p.update_progress = progress_updater;
    return WritePointCloud(filename, pointcloud, p);
}

}  // namespace io
}  // namespace open3d
