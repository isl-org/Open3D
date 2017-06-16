// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Qianyi Zhou <Qianyi.Zhou@gmail.com>
//                    Jaesik Park <syncel@gmail.com>
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

class Image : public Geometry2D
{
public:
	Image() : Geometry2D(GEOMETRY_IMAGE) {};
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

	float FloatValueAtUnsafe(int u, int v) {
		return *((float *)(data_.data() + (u + v * width_) * bytes_per_channel_));
	}

	std::pair<bool, double> FloatValueAt(double u, double v);
		
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

enum AverageType {
	EQUAL,
	WEIGHTED,
};

/// Return an gray scaled float type image.
std::shared_ptr<Image> CreateFloatImageFromImage(
		const Image &image, AverageType average_type = WEIGHTED);

enum FilterType {
	FILTER_GAUSSIAN_3,
	FILTER_GAUSSIAN_5,
	FILTER_GAUSSIAN_7,
	FILTER_SOBEL_3_DX,
	FILTER_SOBEL_3_DY
};

/// Isotropic 2D kernels are seperable: 
/// two 1D kernels are applied in x and y direction.
const std::vector<double> Gaussian3 =
		{ 0.25, 0.5, 0.25 };
const std::vector<double> Gaussian5 =
		{ 0.0625, 0.25, 0.375, 0.25, 0.0625 };
const std::vector<double> Gaussian7 =
		{ 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 };
const std::vector<double> Sobel31 =
		{ -1.0, 0.0, 1.0 };
const std::vector<double> Sobel32 =
		{ 1.0, 2.0, 1.0 };

template<typename T>
T *PointerAt(const Image &image, int u, int v) {
	return (T *)(image.data_.data() +
			(v * image.width_ + u) * sizeof(T));
}

template<typename T>
T *PointerAt(const Image &image, int u, int v, int ch) {
	return (T *)(image.data_.data() +
			((v * image.width_ + u) * image.num_of_channels_ + ch) * sizeof(T));
}

std::shared_ptr<Image> ConvertDepthToFloatImage(const Image &depth, 
		double depth_scale = 1000.0, double depth_trunc = 3.0);

std::shared_ptr<Image> FlipImage(const Image &input);

/// Function to filter image with pre-defined filtering type
std::shared_ptr<Image> FilterImage(const Image &input, FilterType type);

/// Function to filter image with arbitrary dx, dy separable filters
std::shared_ptr<Image> FilterImage(const Image &input,
		const std::vector<double> dx, const std::vector<double> dy);

std::shared_ptr<Image> FilterHorizontalImage(
		const Image &input, const std::vector<double> &kernel);

/// Function to 2x image downsample using simple 2x2 averaging
std::shared_ptr<Image> DownsampleImage(const Image &input);

/// Function to linearly transform pixel intensities
/// image_new = scale * image + offset
void LinearTransformImage(Image &input, double scale = 1.0, double offset = 0.0);

/// Function to cilpping pixel intensities
/// min is lower bound
/// max is upper bound
void ClipIntensityImage(Image &input, double min = 0.0, double max = 1.0);

/// Function to change data types of image
/// crafted for specific usage such as
/// single channel float image -> 8-bit RGB or 16-bit depth image
template <typename T>
std::shared_ptr<Image> CreateImageFromFloatImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
			input.bytes_per_channel_ != 4) {
		PrintDebug("[TypecastImage] Unsupported image format.\n");
		return output;
	}

	output->PrepareImage(
			input.width_, input.height_, input.num_of_channels_, sizeof(T));
	const float *pi = (const float *)input.data_.data();
	T *p = (T*)output->data_.data();
	for (int i = 0; i < input.height_ * input.width_; i++, p++, pi++) {
		if (sizeof(T) == 1)
			*p = static_cast<T>(*pi * 255.0f);
		if (sizeof(T) == 2) 
			*p = static_cast<T>(*pi);
	}
	return output;
}

/// Typedef and functions for ImagePyramid
typedef std::vector<std::shared_ptr<Image>> ImagePyramid;

ImagePyramid FilterImagePyramid(const ImagePyramid &input, const FilterType type);

ImagePyramid CreateImagePyramid(const Image& image, 
		const size_t num_of_levels, const bool with_gaussian_filter = true);

typedef std::vector<std::shared_ptr<Image>> ImagePyramid;

}	// namespace three
