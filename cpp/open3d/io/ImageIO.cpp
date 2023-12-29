// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/ImageIO.h"

#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Image &)>>
        file_extension_to_image_read_function{
                {"png", ReadImageFromPNG},
                {"jpg", ReadImageFromJPG},
                {"jpeg", ReadImageFromJPG},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Image &, int)>>
        file_extension_to_image_write_function{
                {"png", WriteImageToPNG},
                {"jpg", WriteImageToJPG},
                {"jpeg", WriteImageToJPG},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Image failed: missing file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_read_function.find(filename_ext);
    if (map_itr == file_extension_to_image_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Image failed: file extension {} unknown",
                filename_ext);
        return false;
    }
    return map_itr->second(filename, image);
}

bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality /* = kOpen3DImageIODefaultQuality*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_write_function.find(filename_ext);
    if (map_itr == file_extension_to_image_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image, quality);
}

std::shared_ptr<geometry::Image> CreateImageFromMemory(
        const std::string &image_format,
        const unsigned char *image_data_ptr,
        size_t image_data_size) {
    auto image = std::make_shared<geometry::Image>();
    ReadImageFromMemory(image_format, image_data_ptr, image_data_size, *image);
    return image;
}

bool ReadImageFromMemory(const std::string &image_format,
                         const unsigned char *image_data_ptr,
                         size_t image_data_size,
                         geometry::Image &image) {
    if (image_format == "png") {
        return ReadPNGFromMemory(image_data_ptr, image_data_size, image);
    } else if (image_format == "jpg") {
        return ReadJPGFromMemory(image_data_ptr, image_data_size, image);
    } else {
        utility::LogWarning("The format of {} is not supported", image_format);
        return false;
    }
}

}  // namespace io
}  // namespace open3d
