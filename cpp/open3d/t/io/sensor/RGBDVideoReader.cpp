// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/sensor/RGBDVideoReader.h"

#include <string>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/t/io/sensor/realsense/RSBagReader.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace t {
namespace io {

std::string RGBDVideoReader::ToString() const {
    if (IsOpened()) {
        return fmt::format(
                "RGBDVideoReader reading file {} at position {}us / {}us",
                GetFilename(), GetTimestamp(),
                GetMetadata().stream_length_usec_);
    } else {
        return "RGBDVideoReader: No open file.";
    }
}

void RGBDVideoReader::SaveFrames(const std::string &frame_path,
                                 uint64_t start_time,
                                 uint64_t end_time) {
    if (!IsOpened()) {
        utility::LogError("Null file handler. Please call Open().");
    }
    bool success = utility::filesystem::MakeDirectoryHierarchy(
            fmt::format("{}/color", frame_path));
    success &= utility::filesystem::MakeDirectoryHierarchy(
            fmt::format("{}/depth", frame_path));
    if (!success) {
        utility::LogError(
                "Could not create color or depth subfolder in {} or they "
                "already exist.",
                frame_path);
    }
    open3d::io::WriteIJsonConvertibleToJSON(
            fmt::format("{}/intrinsic.json", frame_path), GetMetadata());
    SeekTimestamp(start_time);
    int idx = 0;
    open3d::geometry::Image im_color, im_depth;
    for (auto tim_rgbd = NextFrame(); !IsEOF() && GetTimestamp() < end_time;
         ++idx, tim_rgbd = NextFrame())
#pragma omp parallel sections
    {
#pragma omp section
        {
            im_color = tim_rgbd.color_.ToLegacy();
            auto color_file =
                    fmt::format("{0}/color/{1:05d}.jpg", frame_path, idx);
            open3d::io::WriteImage(color_file, im_color);
            utility::LogDebug("Written color image to {}", color_file);
        }
#pragma omp section
        {
            im_depth = tim_rgbd.depth_.ToLegacy();
            auto depth_file =
                    fmt::format("{0}/depth/{1:05d}.png", frame_path, idx);
            open3d::io::WriteImage(depth_file, im_depth);
            utility::LogDebug("Written depth image to {}", depth_file);
        }
    }
    utility::LogInfo("Written {} depth and color images to {}/{{depth,color}}/",
                     idx, frame_path);
}

std::unique_ptr<RGBDVideoReader> RGBDVideoReader::Create(
        const std::string &filename) {
#ifdef BUILD_LIBREALSENSE
    if (utility::ToLower(filename).compare(filename.length() - 4, 4, ".bag") ==
        0) {
        auto reader = std::make_unique<RSBagReader>();
        reader->Open(filename);
        return reader;
    } else {
        utility::LogError("Unsupported file format for {}.", filename);
    }
#endif
    utility::LogError(
            "Build with -DBUILD_LIBREALSENSE=ON to use RealSense bag file.");
}

}  // namespace io
}  // namespace t
}  // namespace open3d
