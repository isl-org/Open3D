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

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Open3D/Geometry/Geometry2D.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;

/// Typedef and functions for ImagePyramid
typedef std::vector<std::shared_ptr<Image>> ImagePyramid;

class Image : public Geometry2D {
public:
    enum class ColorToIntensityConversionType {
        Equal,
        Weighted,
    };

    enum class FilterType {
        Gaussian3,
        Gaussian5,
        Gaussian7,
        Sobel3Dx,
        Sobel3Dy
    };

public:
    Image() : Geometry2D(Geometry::GeometryType::Image) {}
    ~Image() override {}

public:
    Image &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2d GetMinBound() const override;
    Eigen::Vector2d GetMaxBound() const override;
    bool TestImageBoundary(double u, double v, double inner_margin = 0.0) const;

public:
    virtual bool HasData() const {
        return width_ > 0 && height_ > 0 &&
               data_.size() == size_t(height_ * BytesPerLine());
    }

    Image &Prepare(int width,
                   int height,
                   int num_of_channels,
                   int bytes_per_channel) {
        width_ = width;
        height_ = height;
        num_of_channels_ = num_of_channels;
        bytes_per_channel_ = bytes_per_channel;
        AllocateDataBuffer();
        return *this;
    }

    int BytesPerLine() const {
        return width_ * num_of_channels_ * bytes_per_channel_;
    }

    /// Function to access the bilinear interpolated float value of a
    /// (single-channel) float image.
    /// Returns a tuple, where the first bool indicates if the u,v coordinates
    /// are withing the image dimensions, and the second double value is the
    /// interpolated pixel value.
    std::pair<bool, double> FloatValueAt(double u, double v) const;

    /// Factory function to create a float image composed of multipliers that
    /// convert depth values into camera distances (ImageFactory.cpp)
    /// The multiplier function M(u,v) is defined as:
    /// M(u, v) = sqrt(1 + ((u - cx) / fx) ^ 2 + ((v - cy) / fy) ^ 2)
    /// This function is used as a convenient function for performance
    /// optimization in volumetric integration (see
    /// Core/Integration/TSDFVolume.h).
    static std::shared_ptr<Image>
    CreateDepthToCameraDistanceMultiplierFloatImage(
            const camera::PinholeCameraIntrinsic &intrinsic);

    /// Return a gray scaled float type image.
    std::shared_ptr<Image> CreateFloatImage(
            Image::ColorToIntensityConversionType type =
                    Image::ColorToIntensityConversionType::Weighted) const;

    /// Function to access the raw data of a single-channel Image
    template <typename T>
    T *PointerAt(int u, int v) const;

    /// Function to access the raw data of a multi-channel Image
    template <typename T>
    T *PointerAt(int u, int v, int ch) const;

    std::shared_ptr<Image> ConvertDepthToFloatImage(
            double depth_scale = 1000.0, double depth_trunc = 3.0) const;

    std::shared_ptr<Image> Flip() const;

    /// Function to filter image with pre-defined filtering type
    std::shared_ptr<Image> Filter(Image::FilterType type) const;

    /// Function to filter image with arbitrary dx, dy separable filters
    std::shared_ptr<Image> Filter(const std::vector<double> &dx,
                                  const std::vector<double> &dy) const;

    std::shared_ptr<Image> FilterHorizontal(
            const std::vector<double> &kernel) const;

    /// Function to 2x image downsample using simple 2x2 averaging
    std::shared_ptr<Image> Downsample() const;

    /// Function to dilate 8bit mask map
    std::shared_ptr<Image> Dilate(int half_kernel_size = 1) const;

    /// Function to linearly transform pixel intensities
    /// image_new = scale * image + offset
    Image &LinearTransform(double scale = 1.0, double offset = 0.0);

    /// Function to clipping pixel intensities
    /// min is lower bound
    /// max is upper bound
    Image &ClipIntensity(double min = 0.0, double max = 1.0);

    /// Function to change data types of image
    /// crafted for specific usage such as
    /// single channel float image -> 8-bit RGB or 16-bit depth image
    template <typename T>
    std::shared_ptr<Image> CreateImageFromFloatImage() const;

    /// Function to filter image pyramid
    static ImagePyramid FilterPyramid(const ImagePyramid &input,
                                      Image::FilterType type);

    /// Function to create image pyramid
    ImagePyramid CreatePyramid(size_t num_of_levels,
                               bool with_gaussian_filter = true) const;

    /// Function to create a depthmap boundary mask from depth image
    std::shared_ptr<Image> CreateDepthBoundaryMask(
            double depth_threshold_for_discontinuity_check = 0.1,
            int half_dilation_kernel_size_for_discontinuity_map = 3) const;

protected:
    void AllocateDataBuffer() {
        data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
    }

public:
    int width_ = 0;
    int height_ = 0;
    int num_of_channels_ = 0;
    int bytes_per_channel_ = 0;
    std::vector<uint8_t> data_;
};

}  // namespace geometry
}  // namespace open3d
