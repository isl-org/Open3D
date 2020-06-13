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

RGBDImage &RGBDImage::Clear() {
    color_.Clear();
    depth_.Clear();
    return *this;
}

bool RGBDImage::IsEmpty() const {
    return !color_.HasData() || !depth_.HasData();
}

Eigen::Vector2d RGBDImage::GetMinBound() const {
    return Eigen::Vector2d(0.0, 0.0);
}

Eigen::Vector2d RGBDImage::GetMaxBound() const {
    return Eigen::Vector2d(color_.width_ + depth_.width_, color_.height_);
}

RGBDImagePyramid RGBDImage::FilterPyramid(
        const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type) {
    RGBDImagePyramid rgbd_image_pyramid_filtered;
    rgbd_image_pyramid_filtered.clear();
    int num_of_levels = (int)rgbd_image_pyramid.size();
    for (int level = 0; level < num_of_levels; level++) {
        auto color_level = rgbd_image_pyramid[level]->color_;
        auto depth_level = rgbd_image_pyramid[level]->depth_;
        auto color_level_filtered = color_level.Filter(type);
        auto depth_level_filtered = depth_level.Filter(type);
        auto rgbd_image_level_filtered = std::make_shared<RGBDImage>(
                RGBDImage(*color_level_filtered, *depth_level_filtered));
        rgbd_image_pyramid_filtered.push_back(rgbd_image_level_filtered);
    }
    return rgbd_image_pyramid_filtered;
}

RGBDImagePyramid RGBDImage::CreatePyramid(
        size_t num_of_levels,
        bool with_gaussian_filter_for_color /* = true */,
        bool with_gaussian_filter_for_depth /* = false */) const {
    ImagePyramid color_pyramid =
            color_.CreatePyramid(num_of_levels, with_gaussian_filter_for_color);
    ImagePyramid depth_pyramid =
            depth_.CreatePyramid(num_of_levels, with_gaussian_filter_for_depth);
    RGBDImagePyramid rgbd_image_pyramid;
    rgbd_image_pyramid.clear();
    for (size_t level = 0; level < num_of_levels; level++) {
        auto rgbd_image_level = std::make_shared<RGBDImage>(
                RGBDImage(*color_pyramid[level], *depth_pyramid[level]));
        rgbd_image_pyramid.push_back(rgbd_image_level);
    }
    return rgbd_image_pyramid;
}

}  // namespace geometry
}  // namespace open3d
