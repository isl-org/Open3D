// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <string>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"
#include "torch/script.h"

//
// helper functions for parsing arguments
//

inline open3d::ml::impl::CoordinateMapping ParseCoordinateMappingStr(
        const std::string& str) {
    using open3d::ml::impl::CoordinateMapping;
    CoordinateMapping coordinate_mapping =
            CoordinateMapping::BALL_TO_CUBE_RADIAL;
    if (str == "ball_to_cube_radial") {
        coordinate_mapping = CoordinateMapping::BALL_TO_CUBE_RADIAL;
    } else if (str == "ball_to_cube_volume_preserving") {
        coordinate_mapping = CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING;
    } else if (str == "identity") {
        coordinate_mapping = CoordinateMapping::IDENTITY;
    } else {
        TORCH_CHECK(false,
                    "coordinate_mapping must be one of ('ball_to_cube_radial', "
                    "'ball_to_cube_volume_preserving', 'identity') but got " +
                            str);
    }
    return coordinate_mapping;
}

inline open3d::ml::impl::InterpolationMode ParseInterpolationStr(
        const std::string& str) {
    using open3d::ml::impl::InterpolationMode;
    InterpolationMode interpolation = InterpolationMode::LINEAR;
    if (str == "linear") {
        interpolation = InterpolationMode::LINEAR;
    } else if (str == "linear_border") {
        interpolation = InterpolationMode::LINEAR_BORDER;
    } else if (str == "nearest_neighbor") {
        interpolation = InterpolationMode::NEAREST_NEIGHBOR;
    } else {
        TORCH_CHECK(false,
                    "interpolation must be one of ('linear', "
                    "'linear_border', 'nearest_neighbor') but got " +
                            str);
    }
    return interpolation;
}
