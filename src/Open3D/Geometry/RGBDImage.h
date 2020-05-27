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

#pragma once

#include "Open3D/Geometry/Geometry2D.h"
#include "Open3D/Geometry/Image.h"

namespace open3d {
namespace geometry {

class RGBDImage;

/// Typedef and functions for RGBDImagePyramid
typedef std::vector<std::shared_ptr<RGBDImage>> RGBDImagePyramid;

/// \class RGBDImage
///
/// \brief RGBDImage is for a pair of registered color and depth images,
///
/// viewed from the same view, of the same resolution.
/// If you have other format, convert it first.
class RGBDImage : public Geometry2D {
public:
    /// \brief Default Comnstructor.
    RGBDImage() : Geometry2D(Geometry::GeometryType::RGBDImage) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    RGBDImage(const Image &color, const Image &depth)
        : Geometry2D(Geometry::GeometryType::RGBDImage),
          color_(color),
          depth_(depth) {}

    ~RGBDImage() override {
        color_.Clear();
        depth_.Clear();
    };

    RGBDImage &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2d GetMinBound() const override;
    Eigen::Vector2d GetMaxBound() const override;

    /// \brief Factory function to create an RGBD Image from color and depth
    /// Images.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param depth_scale The ratio to scale depth values. The depth values
    /// will first be scaled and then truncated. \param depth_trunc Depth values
    /// larger than depth_trunc gets truncated to 0. The depth values will first
    /// be scaled and then truncated. \param convert_rgb_to_intensity - Whether
    /// to convert RGB image to intensity image.
    static std::shared_ptr<RGBDImage> CreateFromColorAndDepth(
            const Image &color,
            const Image &depth,
            double depth_scale = 1000.0,
            double depth_trunc = 3.0,
            bool convert_rgb_to_intensity = true);

    /// \brief Factory function to create an RGBD Image from Redwood dataset.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param convert_rgb_to_intensity Whether to convert RGB image to
    /// intensity image.
    static std::shared_ptr<RGBDImage> CreateFromRedwoodFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// \brief Factory function to create an RGBD Image from TUM dataset.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param convert_rgb_to_intensity Whether to convert RGB image to
    /// intensity image.
    static std::shared_ptr<RGBDImage> CreateFromTUMFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// \brief Factory function to create an RGBD Image from SUN3D dataset.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param convert_rgb_to_intensity Whether to convert RGB image to
    /// intensity image.
    static std::shared_ptr<RGBDImage> CreateFromSUNFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// \brief Factory function to create an RGBD Image from NYU dataset.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param convert_rgb_to_intensity Whether to convert RGB image to
    /// intensity image.
    static std::shared_ptr<RGBDImage> CreateFromNYUFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    static RGBDImagePyramid FilterPyramid(
            const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type);

    RGBDImagePyramid CreatePyramid(
            size_t num_of_levels,
            bool with_gaussian_filter_for_color = true,
            bool with_gaussian_filter_for_depth = false) const;

public:
    /// The color image.
    Image color_;
    /// The depth image.
    Image depth_;
};

}  // namespace geometry
}  // namespace open3d
