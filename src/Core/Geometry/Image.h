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

#include <vector>
#include <memory>
#include <Eigen/Core>

#include <Core/Geometry/Geometry2D.h>
#include <Core/Utility/Console.h>

namespace three {

class PinholeCameraIntrinsic;

class Image : public Geometry2D
{
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
	Image() : Geometry2D(Geometry::GeometryType::Image) {};
	~Image() override {};

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector2d GetMinBound() const override;
	Eigen::Vector2d GetMaxBound() const override;

public:
	virtual bool HasData() const {
		return width_ > 0 && height_ > 0 &&
				data_.size() == height_ * BytesPerLine();
	}

	void PrepareImage(int width, int height, int num_of_channels,
			int bytes_per_channel) {
		width_ = width;
		height_ = height;
		num_of_channels_ = num_of_channels;
		bytes_per_channel_ = bytes_per_channel;
		AllocateDataBuffer();
	}

	int BytesPerLine() const {
		return width_ * num_of_channels_ * bytes_per_channel_;
	}

	/// Function to access the bilinear interpolated float value of a
	/// (single-channel) float image
	std::pair<bool, double> FloatValueAt(double u, double v) const;

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

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<Image> CreateImageFromFile(const std::string &filename);

/// Factory function to create a float image composed of multipliers that
/// convert depth values into camera distances (ImageFactory.cpp)
/// The multiplier function M(u,v) is defined as:
/// M(u, v) = sqrt(1 + ((u - cx) / fx) ^ 2 + ((v - cy) / fy) ^ 2)
/// This function is used as a convenient function for performance optimization
/// in volumetric integration (see Core/Integration/TSDFVolume.h).
std::shared_ptr<Image> CreateDepthToCameraDistanceMultiplierFloatImage(
		const PinholeCameraIntrinsic &intrinsic);

/// Return a gray scaled float type image.
std::shared_ptr<Image> CreateFloatImageFromImage(
		const Image &image,
		Image::ColorToIntensityConversionType type =
				Image::ColorToIntensityConversionType::Weighted);

/// Function to access the raw data of a single-channel Image
template<typename T>
T *PointerAt(const Image &image, int u, int v);

/// Function to access the raw data of a multi-channel Image
template<typename T>
T *PointerAt(const Image &image, int u, int v, int ch);

std::shared_ptr<Image> ConvertDepthToFloatImage(const Image &depth,
		double depth_scale = 1000.0, double depth_trunc = 3.0);

std::shared_ptr<Image> FlipImage(const Image &input);

/// Function to filter image with pre-defined filtering type
std::shared_ptr<Image> FilterImage(const Image &input, Image::FilterType type);

/// Function to filter image with arbitrary dx, dy separable filters
std::shared_ptr<Image> FilterImage(const Image &input,
		const std::vector<double> &dx, const std::vector<double> &dy);

std::shared_ptr<Image> FilterHorizontalImage(
		const Image &input, const std::vector<double> &kernel);

/// Function to 2x image downsample using simple 2x2 averaging
std::shared_ptr<Image> DownsampleImage(const Image &input);

/// Function to linearly transform pixel intensities
/// image_new = scale * image + offset
void LinearTransformImage(Image &input,
		double scale = 1.0, double offset = 0.0);

/// Function to clipping pixel intensities
/// min is lower bound
/// max is upper bound
void ClipIntensityImage(Image &input, double min = 0.0, double max = 1.0);

/// Function to change data types of image
/// crafted for specific usage such as
/// single channel float image -> 8-bit RGB or 16-bit depth image
template <typename T>
std::shared_ptr<Image> CreateImageFromFloatImage(const Image &input);

/// Typedef and functions for ImagePyramid
typedef std::vector<std::shared_ptr<Image>> ImagePyramid;

ImagePyramid FilterImagePyramid(const ImagePyramid &input,
		Image::FilterType type);

ImagePyramid CreateImagePyramid(const Image& image,
		size_t num_of_levels, bool with_gaussian_filter = true);

}	// namespace three
