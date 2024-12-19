// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>  // Include after cstddef to define size_t
// clang-format on

#include "open3d/io/ImageIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace io {
namespace {

/// Convert libjpeg error messages to std::runtime_error. This prevents
/// libjpeg from exit() in case of errors.
void jpeg_error_throw(j_common_ptr p_cinfo) {
    if (p_cinfo->is_decompressor)
        jpeg_destroy_decompress(
                reinterpret_cast<jpeg_decompress_struct *>(p_cinfo));
    else
        jpeg_destroy_compress(
                reinterpret_cast<jpeg_compress_struct *>(p_cinfo));
    char buffer[JMSG_LENGTH_MAX];
    (*p_cinfo->err->format_message)(p_cinfo, buffer);
    throw std::runtime_error(buffer);
}

}  // namespace

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file_in;
    JSAMPARRAY buffer;

    if ((file_in = utility::filesystem::FOpen(filename, "rb")) == NULL) {
        utility::LogWarning("Read JPG failed: unable to open file: {}",
                            filename);
        image.Clear();
        return false;
    }

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, file_in);
        jpeg_read_header(&cinfo, TRUE);

        // We only support two channel types: gray, and RGB.
        int num_of_channels = 3;
        int bytes_per_channel = 1;
        switch (cinfo.jpeg_color_space) {
            case JCS_RGB:
            case JCS_YCbCr:
                cinfo.out_color_space = JCS_RGB;
                cinfo.out_color_components = 3;
                num_of_channels = 3;
                break;
            case JCS_GRAYSCALE:
                cinfo.jpeg_color_space = JCS_GRAYSCALE;
                cinfo.out_color_components = 1;
                num_of_channels = 1;
                break;
            case JCS_CMYK:
            case JCS_YCCK:
            default:
                utility::LogWarning(
                        "Read JPG failed: color space not supported.");
                jpeg_destroy_decompress(&cinfo);
                fclose(file_in);
                image.Clear();
                return false;
        }
        jpeg_start_decompress(&cinfo);
        image.Prepare(cinfo.output_width, cinfo.output_height, num_of_channels,
                      bytes_per_channel);
        int row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                            row_stride, 1);
        uint8_t *pdata = image.data_.data();
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(pdata, buffer[0], row_stride);
            pdata += row_stride;
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(file_in);
        return true;
    } catch (const std::runtime_error &err) {
        fclose(file_in);
        image.Clear();
        utility::LogWarning("libjpeg error: {}", err.what());
        return false;
    }
}

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality /* = kOpen3DImageIODefaultQuality*/) {
    if (!image.HasData()) {
        utility::LogWarning("Write JPG failed: image has no data.");
        return false;
    }
    if (image.bytes_per_channel_ != 1 ||
        (image.num_of_channels_ != 1 && image.num_of_channels_ != 3)) {
        utility::LogWarning("Write JPG failed: unsupported image data.");
        return false;
    }
    if (quality == kOpen3DImageIODefaultQuality)  // Set default quality
        quality = 90;
    if (quality < 0 || quality > 100) {
        utility::LogWarning(
                "Write JPG failed: image quality should be in the range "
                "[0,100].");
        return false;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file_out;
    JSAMPROW row_pointer[1];

    if ((file_out = utility::filesystem::FOpen(filename, "wb")) == NULL) {
        utility::LogWarning("Write JPG failed: unable to open file: {}",
                            filename);
        return false;
    }

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, file_out);
        cinfo.image_width = image.width_;
        cinfo.image_height = image.height_;
        cinfo.input_components = image.num_of_channels_;
        cinfo.in_color_space =
                (cinfo.input_components == 1 ? JCS_GRAYSCALE : JCS_RGB);
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality, TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        int row_stride = image.width_ * image.num_of_channels_;
        const uint8_t *pdata = image.data_.data();
        std::vector<uint8_t> buffer(row_stride);
        while (cinfo.next_scanline < cinfo.image_height) {
            memcpy(buffer.data(), pdata, row_stride);
            row_pointer[0] = buffer.data();
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
            pdata += row_stride;
        }
        jpeg_finish_compress(&cinfo);
        fclose(file_out);
        jpeg_destroy_compress(&cinfo);
        return true;
    } catch (const std::runtime_error &err) {
        fclose(file_out);
        utility::LogWarning(err.what());
        return false;
    }
}

bool ReadJPGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPARRAY buffer;

    try {
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = jpeg_error_throw;
        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, image_data_ptr, image_data_size);
        jpeg_read_header(&cinfo, TRUE);

        // We only support two channel types: gray, and RGB.
        int num_of_channels = 3;
        int bytes_per_channel = 1;
        switch (cinfo.jpeg_color_space) {
            case JCS_RGB:
            case JCS_YCbCr:
                cinfo.out_color_space = JCS_RGB;
                cinfo.out_color_components = 3;
                num_of_channels = 3;
                break;
            case JCS_GRAYSCALE:
                cinfo.jpeg_color_space = JCS_GRAYSCALE;
                cinfo.out_color_components = 1;
                num_of_channels = 1;
                break;
            case JCS_CMYK:
            case JCS_YCCK:
            default:
                utility::LogWarning(
                        "Read JPG failed: color space not supported.");
                jpeg_destroy_decompress(&cinfo);
                image.Clear();
                return false;
        }
        jpeg_start_decompress(&cinfo);
        image.Prepare(cinfo.output_width, cinfo.output_height, num_of_channels,
                      bytes_per_channel);
        int row_stride = cinfo.output_width * cinfo.output_components;
        buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                            row_stride, 1);
        uint8_t *pdata = image.data_.data();
        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(pdata, buffer[0], row_stride);
            pdata += row_stride;
        }
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        return true;
    } catch (const std::runtime_error &err) {
        image.Clear();
        utility::LogWarning(err.what());
        return false;
    }
}

}  // namespace io
}  // namespace open3d
