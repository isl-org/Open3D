// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/camera/PinholeCameraTrajectory.h"

namespace open3d {
namespace io {

/// Factory function to create a PinholeCameraTrajectory from a file
/// (PinholeCameraTrajectoryFactory.cpp)
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<camera::PinholeCameraTrajectory>
CreatePinholeCameraTrajectoryFromFile(const std::string &filename);

/// The general entrance for reading a PinholeCameraTrajectory from a file
/// The function calls read functions based on the extension name of filename.
/// \return If the read function is successful.
bool ReadPinholeCameraTrajectory(const std::string &filename,
                                 camera::PinholeCameraTrajectory &trajectory);

/// The general entrance for writing a PinholeCameraTrajectory to a file
/// The function calls write functions based on the extension name of filename.
/// \return If the write function is successful.
bool WritePinholeCameraTrajectory(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

bool ReadPinholeCameraTrajectoryFromLOG(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory);

bool WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

bool ReadPinholeCameraTrajectoryFromTUM(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory);

bool WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

}  // namespace io
}  // namespace open3d
