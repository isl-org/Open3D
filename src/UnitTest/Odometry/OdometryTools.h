// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Geometry/Image.h"
#include "TestUtility/UnitTest.h"

namespace odometry_tools {
std::shared_ptr<open3d::geometry::Image> GenerateImage(
        const int& width,
        const int& height,
        const int& num_of_channels,
        const int& bytes_per_channel,
        const float& vmin,
        const float& vmax,
        const int& seed);

// Shift the pixels left with a specified step.
void ShiftLeft(std::shared_ptr<open3d::geometry::Image> image, const int& step);

// Shift the pixels up with a specified step.
void ShiftUp(std::shared_ptr<open3d::geometry::Image> image, const int& step);

// Create dummy correspondence map object.
std::shared_ptr<open3d::geometry::Image> CorrespondenceMap(const int& width,
                                                           const int& height,
                                                           const int& vmin,
                                                           const int& vmax,
                                                           const int& seed);

// Create dummy depth buffer object.
std::shared_ptr<open3d::geometry::Image> DepthBuffer(const int& width,
                                                     const int& height,
                                                     const float& vmin,
                                                     const float& vmax,
                                                     const int& seed);
}  // namespace odometry_tools
