// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <png.h>

#include "open3d/io/ImageIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

void SetPNGImageFromImage(const geometry::Image &image,
                          int quality,
                          png_image &pngimage) {
    pngimage.width = image.width_;
    pngimage.height = image.height_;
    pngimage.format = pngimage.flags = 0;

    if (image.bytes_per_channel_ == 2) {
        pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
    }
    if (image.num_of_channels_ >= 3) {
        pngimage.format |= PNG_FORMAT_FLAG_COLOR;
    }
    if (image.num_of_channels_ == 4) {
        pngimage.format |= PNG_FORMAT_FLAG_ALPHA;
    }
    if (quality <= 2) {
        pngimage.flags |= PNG_IMAGE_FLAG_FAST;
    }
}

}  // unnamed namespace

namespace io {

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&pngimage, filename.c_str()) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        return false;
    }

    // Clear colormap flag if necessary to ensure libpng expands the colo
    // indexed pixels to full color
    if (pngimage.format & PNG_FORMAT_FLAG_COLORMAP) {
        pngimage.format &= ~PNG_FORMAT_FLAG_COLORMAP;
    }

    image.Prepare(pngimage.width, pngimage.height,
                  PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format),
                  PNG_IMAGE_SAMPLE_COMPONENT_SIZE(pngimage.format));

    if (png_image_finish_read(&pngimage, NULL, image.data_.data(), 0, NULL) ==
        0) {
        utility::LogWarning("Read PNG failed: unable to read file: {}",
                            filename);
        utility::LogWarning("PNG error: {}", pngimage.message);
        return false;
    }
    return true;
}

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality) {
    if (!image.HasData()) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    if (quality == kOpen3DImageIODefaultQuality)  // Set default quality
        quality = 6;
    if (quality < 0 || quality > 9) {
        utility::LogWarning(
                "Write PNG failed: quality ({}) must be in the range [0,9]",
                quality);
        return false;
    }
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    SetPNGImageFromImage(image, quality, pngimage);
    if (png_image_write_to_file(&pngimage, filename.c_str(), 0,
                                image.data_.data(), 0, NULL) == 0) {
        utility::LogWarning("Write PNG failed: unable to write file: {}",
                            filename);
        return false;
    }
    return true;
}

bool ReadPNGFromMemory(const unsigned char *image_data_ptr,
                       size_t image_data_size,
                       geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_memory(&pngimage, image_data_ptr,
                                         image_data_size) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        return false;
    }

    // Clear colormap flag if necessary to ensure libpng expands the colo
    // indexed pixels to full color
    if (pngimage.format & PNG_FORMAT_FLAG_COLORMAP) {
        pngimage.format &= ~PNG_FORMAT_FLAG_COLORMAP;
    }

    image.Prepare(pngimage.width, pngimage.height,
                  PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format),
                  PNG_IMAGE_SAMPLE_COMPONENT_SIZE(pngimage.format));

    if (png_image_finish_read(&pngimage, NULL, image.data_.data(), 0, NULL) ==
        0) {
        utility::LogWarning("PNG error: {}", pngimage.message);
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace open3d
