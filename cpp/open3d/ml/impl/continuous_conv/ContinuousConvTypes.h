// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace ml {
namespace impl {

/// Interpolation modes
/// LINEAR is a standard trilinear interpolation with coordinate clamping
/// LINEAR_BORDER uses a zero border instead of clamping
/// NEAREST_NEIGHBOR no interpolation, use nearest neighbor
enum class InterpolationMode { LINEAR, LINEAR_BORDER, NEAREST_NEIGHBOR };

/// Coordinate Mapping functions
/// - BALL_TO_CUBE_RADIAL uses radial stretching to map a sphere to
///   a cube.
/// - BALL_TO_CUBE_VOLUME_PRESERVING is using a more expensive volume
///   preserving mapping to map a sphere to a cube.
/// - IDENTITY no mapping is applied to the coordinates.
enum class CoordinateMapping {
    BALL_TO_CUBE_RADIAL,
    BALL_TO_CUBE_VOLUME_PRESERVING,
    IDENTITY
};

}  // namespace impl
}  // namespace ml
}  // namespace open3d
