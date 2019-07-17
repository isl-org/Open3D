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

#include "Open3D/Geometry/RGBDImage.h"

namespace open3d {
namespace geometry {

std::shared_ptr<RGBDImage> RGBDImage::CreateFromColorAndDepth(
        const Image &color,
        const Image &depth,
        double depth_scale /* = 1000.0*/,
        double depth_trunc /* = 3.0*/,
        bool convert_rgb_to_intensity /* = true*/) {
    std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogWarning(
                "[CreateFromColorAndDepth] Unsupported image "
                "format.\n");
        return rgbd_image;
    }
    rgbd_image->depth_ =
            *depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
    rgbd_image->color_ =
            convert_rgb_to_intensity ? *color.CreateFloatImage() : color;
    return rgbd_image;
}

/// Reference: http://redwood-data.org/indoor/
/// File format: http://redwood-data.org/indoor/dataset.html
std::shared_ptr<RGBDImage> RGBDImage::CreateFromRedwoodFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    return CreateFromColorAndDepth(color, depth, 1000.0, 4.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://vision.in.tum.de/data/datasets/rgbd-dataset
/// File format: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
std::shared_ptr<RGBDImage> RGBDImage::CreateFromTUMFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    return CreateFromColorAndDepth(color, depth, 5000.0, 4.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://sun3d.cs.princeton.edu/
/// File format: https://github.com/PrincetonVision/SUN3DCppReader
std::shared_ptr<RGBDImage> RGBDImage::CreateFromSUNFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogWarning(
                "[CreateRGBDImageFromSUNFormat] Unsupported image format.\n");
        return rgbd_image;
    }
    for (int v = 0; v < depth.height_; v++) {
        for (int u = 0; u < depth.width_; u++) {
            uint16_t &d = *depth.PointerAt<uint16_t>(u, v);
            d = (d >> 3) | (d << 13);
        }
    }
    // SUN depth map has long range depth. We set depth_trunc as 7.0
    return CreateFromColorAndDepth(color, depth, 1000.0, 7.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
std::shared_ptr<RGBDImage> RGBDImage::CreateFromNYUFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogWarning(
                "[CreateRGBDImageFromNYUFormat] Unsupported image format.\n");
        return rgbd_image;
    }
    for (int v = 0; v < depth.height_; v++) {
        for (int u = 0; u < depth.width_; u++) {
            uint16_t *d = depth.PointerAt<uint16_t>(u, v);
            uint8_t *p = (uint8_t *)d;
            uint8_t x = *p;
            *p = *(p + 1);
            *(p + 1) = x;
            double xx = 351.3 / (1092.5 - *d);
            if (xx <= 0.0) {
                *d = 0;
            } else {
                *d = (uint16_t)(floor(xx * 1000 + 0.5));
            }
        }
    }
    // NYU depth map has long range depth. We set depth_trunc as 7.0
    return CreateFromColorAndDepth(color, depth, 1000.0, 7.0,
                                   convert_rgb_to_intensity);
}

}  // namespace geometry
}  // namespace open3d
