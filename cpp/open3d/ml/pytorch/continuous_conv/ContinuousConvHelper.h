// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
