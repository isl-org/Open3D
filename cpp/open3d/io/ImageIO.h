// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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

/// Factory function to create an image from memory.
std::shared_ptr<geometry::Image> CreateImageFromMemory(
        const std::string &image_format,
        const unsigned char *image_data_ptr,
        size_t image_data_size);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadImage(const std::string &filename, geometry::Image &image);

/// The general entrance for reading an Image from memory
/// The function calls read functions based on format of image.
/// \param image_format the format of image, "png" or "jpg".
/// \param image_data_ptr the pointer to image data in memory.
/// \param image_data_size the size of image data in memory.
/// \return return true if the read function is successful, false otherwise.
bool ReadImageFromMemory(const std::string &image_format,
                         const unsigned char *image_data_ptr,
                         size_t image_data_size,
                         geometry::Image &image);

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

/// The general entrance for reading an Image from memory
bool ReadPNGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image);

bool ReadJPGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image);

}  // namespace io
}  // namespace open3d
