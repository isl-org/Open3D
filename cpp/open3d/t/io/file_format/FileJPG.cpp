// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>  // Include after cstddef to define size_t
// clang-format on

#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file_in;
    JSAMPARRAY buffer;

    if ((file_in = utility::filesystem::FOpen(filename, "rb")) == NULL) {
        utility::LogWarning("Read JPG failed: unable to open file: {}",
                            filename);
        return false;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file_in);
    jpeg_read_header(&cinfo, TRUE);

    // We only support two channel types: gray, and RGB.
    int num_of_channels = 3;
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
            utility::LogWarning("Read JPG failed: color space not supported.");
            jpeg_destroy_decompress(&cinfo);
            fclose(file_in);
            return false;
    }
    jpeg_start_decompress(&cinfo);
    image.Clear();
    image.Reset(cinfo.output_height, cinfo.output_width, num_of_channels,
                core::UInt8, image.GetDevice());

    int row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                        row_stride, 1);
    uint8_t *pdata = static_cast<uint8_t *>(image.GetDataPtr());

    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        core::MemoryManager::MemcpyFromHost(pdata, image.GetDevice(), buffer[0],
                                            row_stride * 1);
        pdata += row_stride;
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file_in);
    return true;
}

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality /* = kOpen3DImageIODefaultQuality*/) {
    if (image.IsEmpty()) {
        utility::LogWarning("Write JPG failed: image has no data.");
        return false;
    }
    if (image.GetDtype() != core::UInt8 ||
        (image.GetChannels() != 1 && image.GetChannels() != 3)) {
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

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file_out);
    cinfo.image_width = image.GetCols();
    cinfo.image_height = image.GetRows();
    cinfo.input_components = image.GetChannels();
    cinfo.in_color_space =
            (cinfo.input_components == 1 ? JCS_GRAYSCALE : JCS_RGB);
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    int row_stride = image.GetCols() * image.GetChannels();
    const uint8_t *pdata = static_cast<const uint8_t *>(image.GetDataPtr());
    std::vector<uint8_t> buffer(row_stride);
    while (cinfo.next_scanline < cinfo.image_height) {
        core::MemoryManager::MemcpyToHost(buffer.data(), pdata,
                                          image.GetDevice(), row_stride * 1);
        row_pointer[0] = buffer.data();
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
        pdata += row_stride;
    }
    jpeg_finish_compress(&cinfo);
    fclose(file_out);
    jpeg_destroy_compress(&cinfo);
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
