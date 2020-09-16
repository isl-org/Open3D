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

#include <string>

#include "open3d/geometry/Image.h"

namespace open3d {
namespace io {

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadImage(const std::string &filename, geometry::Image &image);

constexpr int kOpen3DImageIODefaultQuality = -1;

/// The general entrance for writing an Image to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports quality, the parameter will be used.
/// Otherwise it will be ignored.
/// \param quality: PNG: [0-9] <=2 fast write for storing intermediate data
///                            >=3 (default) normal write for balanced speed and
///                            file size
///                 JPEG: [0-100] Typically in [70,95]. 90 is default (good
///                 quality).
/// \return return true if the write function is successful, false otherwise.
bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality = kOpen3DImageIODefaultQuality);

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image);

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality = kOpen3DImageIODefaultQuality);

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image);

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality = kOpen3DImageIODefaultQuality);

}  // namespace io
}  // namespace open3d
