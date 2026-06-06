// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <png.h>

#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

namespace {

// libpng fills pngimage.message with text that already includes "libpng error:
// ".
void LogLibPNGError(const png_image &pngimage, const char *fallback) {
    if (pngimage.message[0] != '\0') {
        utility::LogWarning("{}", pngimage.message);
    } else {
        utility::LogWarning("PNG error: {}.", fallback);
    }
}

}  // namespace

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

// Shared setup for a png_image struct from a decoded format descriptor.
static bool FinishReadPNG(png_image &pngimage,
                          geometry::Image &image,
                          const char *source_label) {
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
        const std::string fallback =
                std::string("Read PNG failed from ") + source_label;
        LogLibPNGError(pngimage, fallback.c_str());
        image.Clear();
        return false;
    }
    return true;
}

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&pngimage, filename.c_str()) == 0) {
        LogLibPNGError(pngimage, "Read PNG failed: unable to parse header");
        image.Clear();
        return false;
    }
    return FinishReadPNG(pngimage, image, filename.c_str());
}

bool ReadImageFromPNGInMemory(const uint8_t *data,
                              size_t size,
                              geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_memory(&pngimage, data, size) == 0) {
        LogLibPNGError(pngimage,
                       "Read PNG from memory failed: unable to parse header");
        image.Clear();
        return false;
    }
    return FinishReadPNG(pngimage, image, "<memory>");
}

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality) {
    if (image.IsEmpty()) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    if (image.GetDtype() != core::Bool && image.GetDtype() != core::UInt8 &&
        image.GetDtype() != core::UInt16) {
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
        const std::string fallback =
                std::string("unable to write file: ") + filename;
        LogLibPNGError(pngimage, fallback.c_str());
        return false;
    }
    return true;
}

bool WriteImageToPNGInMemory(std::vector<uint8_t> &buffer,
                             const t::geometry::Image &image,
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

    // Compute bytes required
    size_t mem_bytes = 0;
    if (png_image_write_to_memory(&pngimage, nullptr, &mem_bytes, 0,
                                  image.GetDataPtr(), 0, nullptr) == 0) {
        LogLibPNGError(
                pngimage,
                "Write PNG failed: could not compute in-memory buffer size");
        return false;
    }
    buffer.resize(mem_bytes);
    if (png_image_write_to_memory(&pngimage, &buffer[0], &mem_bytes, 0,
                                  image.GetDataPtr(), 0, nullptr) == 0) {
        LogLibPNGError(pngimage,
                       "Write PNG failed: unable to encode to memory");
        buffer.clear();
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
