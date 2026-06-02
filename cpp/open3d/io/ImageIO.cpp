// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/ImageIO.h"

#include <array>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace io {

namespace {

struct ImageSignatureDecoder {
    std::string signature;
    std::function<bool(const std::string &, geometry::Image &)> read_from_file;
    std::function<bool(const unsigned char *, size_t, geometry::Image &)>
            read_from_memory;
};

static const std::array<ImageSignatureDecoder, 2> kImageSignatureDecoders{{
        {"\x89\x50\x4e\x47\xd\xa\x1a\xa", ReadImageFromPNG, ReadPNGFromMemory},
        {"\xFF\xD8\xFF", ReadImageFromJPG, ReadJPGFromMemory},
}};
static constexpr uint8_t MAX_SIGNATURE_LEN = 8;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Image &, int)>>
        file_extension_to_image_write_function{
                {"png", WriteImageToPNG},
                {"jpg", WriteImageToJPG},
                {"jpeg", WriteImageToJPG},
        };
}  // unnamed namespace

std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string signature_buffer(MAX_SIGNATURE_LEN, 0);
    std::ifstream file(filename, std::ios::binary);
    file.read(&signature_buffer[0], MAX_SIGNATURE_LEN);
    std::string err_msg;
    if (!file) {
        err_msg = "Read geometry::Image failed for file {}. I/O error.";
    } else {
        file.close();
        for (const auto &decoder : kImageSignatureDecoders) {
            if (signature_buffer.compare(0, decoder.signature.size(),
                                         decoder.signature) == 0) {
                return decoder.read_from_file(filename, image);
            }
        }
        err_msg =
                "Read geometry::Image failed for file {}. Unknown file "
                "signature, only PNG and JPG are supported.";
    }
    image.Clear();
    utility::LogWarning(err_msg.c_str(), filename);
    return false;
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
                "Write geometry::Image failed: file extension {} unknown.",
                filename_ext);
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
    (void)image_format;
    if (image_data_ptr == nullptr || image_data_size == 0) {
        utility::LogWarning(
                "Read image from memory failed: null or empty buffer.");
        image.Clear();
        return false;
    }
    for (const auto &decoder : kImageSignatureDecoders) {
        const auto &magic = decoder.signature;
        if (image_data_size >= magic.size() &&
            std::memcmp(image_data_ptr, magic.data(), magic.size()) == 0) {
            return decoder.read_from_memory(image_data_ptr, image_data_size,
                                            image);
        }
    }
    utility::LogWarning(
            "Read image from memory failed: unknown file signature, only PNG "
            "and JPG are supported.");
    image.Clear();
    return false;
}

}  // namespace io
}  // namespace open3d
