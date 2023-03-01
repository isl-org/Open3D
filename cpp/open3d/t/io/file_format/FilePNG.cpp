// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <png.h>

#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

static void SetPNGImageFromImage(const geometry::Image &image,
                                 int quality,
                                 png_image &pngimage) {
    pngimage.width = image.GetCols();
    pngimage.height = image.GetRows();
    pngimage.format = pngimage.flags = 0;

    if (image.GetDtype() == core::UInt16) {
        pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
    }
    if (image.GetChannels() >= 3) {
        pngimage.format |= PNG_FORMAT_FLAG_COLOR;
    }
    if (image.GetChannels() == 4) {
        pngimage.format |= PNG_FORMAT_FLAG_ALPHA;
    }
    if (quality <= 2) {
        pngimage.flags |= PNG_IMAGE_FLAG_FAST;
    }
}

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&pngimage, filename.c_str()) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        return false;
    }

    // Clear colormap flag if necessary to ensure libpng expands the color
    // indexed pixels to full color
    if (pngimage.format & PNG_FORMAT_FLAG_COLORMAP) {
        pngimage.format &= ~PNG_FORMAT_FLAG_COLORMAP;
    }
    if (pngimage.format & PNG_FORMAT_FLAG_LINEAR) {
        image.Reset(pngimage.height, pngimage.width,
                    PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format), core::UInt16,
                    image.GetDevice());
    } else {
        image.Reset(pngimage.height, pngimage.width,
                    PNG_IMAGE_SAMPLE_CHANNELS(pngimage.format), core::UInt8,
                    image.GetDevice());
    }

    if (png_image_finish_read(&pngimage, NULL, image.GetDataPtr(), 0, NULL) ==
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
    if (image.IsEmpty()) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    if (image.GetDtype() != core::UInt8 && image.GetDtype() != core::UInt16) {
        utility::LogWarning("Write PNG failed: unsupported image data.");
        return false;
    }
    if (quality == kOpen3DImageIODefaultQuality)  // Set default quality
    {
        quality = 6;
    }
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
                                image.GetDataPtr(), 0, NULL) == 0) {
        utility::LogWarning("Write PNG failed: unable to write file: {}",
                            filename);
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
