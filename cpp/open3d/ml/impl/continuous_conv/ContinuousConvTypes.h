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
