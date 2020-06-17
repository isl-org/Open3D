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

#include <memory>
#include <string>

#include "Open3D/ColorMap/ImageWarpingField.h"

namespace open3d {

namespace color_map {
class ImageWarpingField;
}

namespace io {

/// Factory function to create a ImageWarpingField from a file
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<color_map::ImageWarpingField> CreateImageWarpingFieldFromFile(
        const std::string &filename);

/// The general entrance for reading a ImageWarpingField from a file
/// \return If the read function is successful.
bool ReadImageWarpingField(const std::string &filename,
                           color_map::ImageWarpingField &warping_field);

/// The general entrance for writing a ImageWarpingField to a file
/// \return If the write function is successful.
bool WriteImageWarpingField(const std::string &filename,
                            const color_map::ImageWarpingField &warping_field);

}  // namespace io
}  // namespace open3d
