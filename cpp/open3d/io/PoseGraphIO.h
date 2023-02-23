// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/pipelines/registration/PoseGraph.h"

namespace open3d {
namespace io {

/// Factory function to create a PoseGraph from a file
/// (PinholeCameraTrajectoryFactory.cpp)
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<pipelines::registration::PoseGraph> CreatePoseGraphFromFile(
        const std::string &filename);

/// The general entrance for reading a PoseGraph from a file.
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadPoseGraph(const std::string &filename,
                   pipelines::registration::PoseGraph &pose_graph);

/// The general entrance for writing a PoseGraph to a file.
/// The function calls write functions based on the extension name of filename.
/// \return return true if the write function is successful, false otherwise.
bool WritePoseGraph(const std::string &filename,
                    const pipelines::registration::PoseGraph &pose_graph);

}  // namespace io
}  // namespace open3d
