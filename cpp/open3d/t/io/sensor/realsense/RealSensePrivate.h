// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// PRIVATE RealSense header for compiling Open3D. Do not #include outside
// Open3D.
#pragma once

#include <librealsense2/rs.hpp>

#include "open3d/io/IJsonConvertibleIO.h"

namespace open3d {
namespace t {
namespace io {

DECLARE_STRINGIFY_ENUM(rs2_stream)
DECLARE_STRINGIFY_ENUM(rs2_format)
DECLARE_STRINGIFY_ENUM(rs2_l500_visual_preset)
DECLARE_STRINGIFY_ENUM(rs2_rs400_visual_preset)
DECLARE_STRINGIFY_ENUM(rs2_sr300_visual_preset)

}  // namespace io
}  // namespace t
}  // namespace open3d
