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
std::shared_ptr<Image> CreateFloatImageFromImage(const Image &image, AverageType average_type = WEIGHTED);

enum FilterType {
	FILTER_GAUSSIAN_3,
	FILTER_GAUSSIAN_5,
	FILTER_GAUSSIAN_7,
	FILTER_SOBEL_HORIZONTAL_3,
	FILTER_SOBEL_VERTICAL_3
};

const std::vector<double> Gaussian =
{ 0.0113, 0.0838, 0.0113,
0.0838, 0.6193, 0.0838,
0.0113, 0.0838, 0.0113 };

// Gaussian filter coefficients
// 2D kernels are seperable as filtering 1D kernel twice.
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

// Sobel filter coefficients
const double divfac = 8.0f; // damping factor
const std::vector<double> Sobel_dx =
{ -1.0f / divfac, 0.0f / divfac, 1.0f / divfac,
-2.0f / divfac, 0.0f / divfac, 2.0f / divfac,
-1.0f / divfac, 0.0f / divfac, 1.0f / divfac };
const std::vector<double> Sobel_dy =
{ -1.0f / divfac, -2.0f / divfac, -1.0f / divfac,
0.0f / divfac, 0.0f / divfac, 0.0f / divfac,
1.0f / divfac, 2.0f / divfac, 1.0f / divfac };

//template<typename T>
//T *PointerAt(const Image &image, int u, int v);
//
////template<typename T>
////const T *PointerAt(const Image &image, int u, int v);
//
//template<typename T>
//T *PointerAt(const Image &image, int u, int v, int ch);

template<typename T>
T *PointerAt(const Image &image, int u, int v) {
	return (T *)(image.data_.data() +
		(v * image.width_ + u) * sizeof(T));
}

//template<typename T>
//const T *PointerAt(const Image &image, int u, int v) {
//	return (const T *)(image.data_.data() +
//		(u + v * image.width_) * sizeof(T));
//}

template<typename T>
T *PointerAt(const Image &image, int u, int v, int ch) {
	return (T *)(image.data_.data() +
		((v * image.width_ + u) * image.num_of_channels_ + ch) * sizeof(T));
}

void ConvertDepthToFloatImage(const Image &depth, Image &depth_f,
	double depth_scale = 1000.0, double depth_trunc = 3.0);

std::shared_ptr<Image> FilpImage(const Image &input);

// 3x3 filtering
// assumes single channel float type image
//std::shared_ptr<Image> FilterImage(const Image &input, const std::vector<double> &kernel);
std::shared_ptr<Image> FilterImage(const Image &input, FilterType type);
std::shared_ptr<Image> FilterHorizontalImage(const Image &input, const std::vector<double> &kernel);

// 2x image downsampling
// assumes float type image
// simple 2x2 averaging
// assumes 2x powered image width and height
// need to double check how we are going to handle invalid depth
std::shared_ptr<Image> DownsampleImage(const Image &input);

std::vector<std::shared_ptr<const Image>> CreateImagePyramid(
	const Image& image,
	size_t num_of_levels);

// assumes float type image as an input
template <typename T>
std::shared_ptr<Image> TypecastImage(const Image &input)
{
	// todo: add sanity check.
	
	auto output = std::make_shared<Image>();
	output->PrepareImage(input.height_, input.width_, input.num_of_channels_, sizeof(T));
	const float *pi = (const float *)input.data_.data();
	T *p = output->data_.data();
	for (int i = 0; i < input.height_ * input.width_; i++, p++, pi++) {
		if (sizeof(T) == 1)
			*p = static_cast<T>(*pi * 255.0f);
		if (sizeof(T) == 2)
			*p = static_cast<T>(*pi * 65535.0f);
	}
	return output;
}

}	// namespace three
