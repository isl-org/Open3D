// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "open3d/pipelines/color_map/ImageWarpingField.h"

namespace open3d {

namespace io {

/// Factory function to create a ImageWarpingField from a file
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<pipelines::color_map::ImageWarpingField>
CreateImageWarpingFieldFromFile(const std::string &filename);

/// The general entrance for reading a ImageWarpingField from a file
/// \return If the read function is successful.
bool ReadImageWarpingField(
        const std::string &filename,
        pipelines::color_map::ImageWarpingField &warping_field);

/// The general entrance for writing a ImageWarpingField to a file
/// \return If the write function is successful.
bool WriteImageWarpingField(
        const std::string &filename,
        const pipelines::color_map::ImageWarpingField &warping_field);

}  // namespace io
}  // namespace open3d
