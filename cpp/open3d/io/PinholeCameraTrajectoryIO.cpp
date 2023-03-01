// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/PinholeCameraTrajectoryIO.h"

#include <unordered_map>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

bool ReadPinholeCameraTrajectoryFromJSON(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory) {
    return ReadIJsonConvertible(filename, trajectory);
}

bool WritePinholeCameraTrajectoryToJSON(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    return WriteIJsonConvertibleToJSON(filename, trajectory);
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           camera::PinholeCameraTrajectory &)>>
        file_extension_to_trajectory_read_function{
                {"log", ReadPinholeCameraTrajectoryFromLOG},
                {"json", ReadPinholeCameraTrajectoryFromJSON},
                {"txt", ReadPinholeCameraTrajectoryFromTUM},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const camera::PinholeCameraTrajectory &)>>
        file_extension_to_trajectory_write_function{
                {"log", WritePinholeCameraTrajectoryToLOG},
                {"json", WritePinholeCameraTrajectoryToJSON},
                {"txt", WritePinholeCameraTrajectoryToTUM},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<camera::PinholeCameraTrajectory>
CreatePinholeCameraTrajectoryFromFile(const std::string &filename) {
    auto trajectory = std::make_shared<camera::PinholeCameraTrajectory>();
    ReadPinholeCameraTrajectory(filename, *trajectory);
    return trajectory;
}

bool ReadPinholeCameraTrajectory(const std::string &filename,
                                 camera::PinholeCameraTrajectory &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trajectory_read_function.find(filename_ext);
    if (map_itr == file_extension_to_trajectory_read_function.end()) {
        utility::LogWarning(
                "Read camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

bool WritePinholeCameraTrajectory(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trajectory_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trajectory_write_function.end()) {
        utility::LogWarning(
                "Write camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

}  // namespace io
}  // namespace open3d
