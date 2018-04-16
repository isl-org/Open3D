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

#include "Image.h"

namespace {
/// Isotropic 2D kernels are separable:
/// two 1D kernels are applied in x and y direction.
const std::vector<double> Gaussian3 = { 0.25, 0.5, 0.25 };
const std::vector<double> Gaussian5 = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
const std::vector<double> Gaussian7 =
		{ 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 };
const std::vector<double> Sobel31 = { -1.0, 0.0, 1.0 };
const std::vector<double> Sobel32 = { 1.0, 2.0, 1.0 };
}	//unnamed namespace

namespace three {

void Image::Clear()
{
	width_ = 0;
	height_ = 0;
	data_.clear();
}

bool Image::IsEmpty() const
{
	return !HasData();
}

Eigen::Vector2d Image::GetMinBound() const
{
	return Eigen::Vector2d(0.0, 0.0);
}

Eigen::Vector2d Image::GetMaxBound() const
{
	return Eigen::Vector2d(width_, height_);
}

std::pair<bool, double> Image::FloatValueAt(double u, double v) const
{
	if ((num_of_channels_ != 1) || (bytes_per_channel_ != 4) ||
		(u < 0.0 || u >(double)(width_ - 1) ||
		v < 0.0 || v >(double)(height_ - 1))) {
		return std::make_pair(false, 0.0);
	}
	int ui = std::max(std::min((int)u, width_ - 2), 0);
	int vi = std::max(std::min((int)v, height_ - 2), 0);
	double pu = u - ui;
	double pv = v - vi;
	float value[4] = {
		*PointerAt<float>(*this, ui, vi),
		*PointerAt<float>(*this, ui, vi + 1),
		*PointerAt<float>(*this, ui + 1, vi),
		*PointerAt<float>(*this, ui + 1, vi + 1)
	};
	return std::make_pair(true,
		(value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
		(value[2] * (1 - pv) + value[3] * pv) * pu);
}

template<typename T>
T *PointerAt(const Image &image, int u, int v) {
	return (T *)(image.data_.data() +
			(v * image.width_ + u) * sizeof(T));
}

template float * PointerAt<float>(const Image &image, int u, int v);
template int * PointerAt<int>(const Image &image, int u, int v);
template uint8_t * PointerAt<uint8_t>(const Image &image, int u, int v);
template uint16_t * PointerAt<uint16_t>(const Image &image, int u, int v);

template<typename T>
T *PointerAt(const Image &image, int u, int v, int ch) {
	return (T *)(image.data_.data() +
			((v * image.width_ + u) * image.num_of_channels_ + ch) * sizeof(T));
}

template float * PointerAt<float>(const Image &image, int u, int v, int ch);
template int * PointerAt<int>(const Image &image, int u, int v, int ch);
template uint8_t * PointerAt<uint8_t>(const Image &image, int u, int v, int ch);
template uint16_t * PointerAt<uint16_t>(const Image &image, int u, int v,
		int ch);

std::shared_ptr<Image> ConvertDepthToFloatImage(const Image &depth,
		double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/)
{
	// don't need warning message about image type
	// as we call CreateFloatImageFromImage
	auto output = CreateFloatImageFromImage(depth);
	for (int y = 0; y < output->height_; y++) {
		for (int x = 0; x < output->width_; x++) {
			float *p = PointerAt<float>(*output, x, y);
			*p /= (float)depth_scale;
			if (*p >= depth_trunc)
				*p = 0.0f;
		}
	}
	return output;
}

void ClipIntensityImage(Image &input, double min/* = 0.0*/,
		double max/* = 1.0*/)
{
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[ClipIntensityImage] Unsupported image format.\n");
		return;
	}
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float *p = PointerAt<float>(input, x, y);
			if (*p > max)
				*p = (float)max;
			if (*p < min)
				*p = (float)min;
		}
	}
}

void LinearTransformImage(Image &input, double scale, double offset/* = 0.0*/)
{
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[LinearTransformImage] Unsupported image format.\n");
		return;
	}
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float *p = PointerAt<float>(input, x, y);
			(*p) = (float)(scale * (*p) + offset);
		}
	}
}

std::shared_ptr<Image> DownsampleImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[DownsampleImage] Unsupported image format.\n");
		return output;
	}
	int half_width = (int)floor((double)input.width_ / 2.0);
	int half_height = (int)floor((double)input.height_ / 2.0);
	output->PrepareImage(half_width, half_height, 1, 4);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int y = 0; y < output->height_; y++) {
		for (int x = 0; x < output->width_; x++) {
			float *p1 = PointerAt<float>(input, x * 2, y * 2);
			float *p2 = PointerAt<float>(input, x * 2 + 1, y * 2);
			float *p3 = PointerAt<float>(input, x * 2, y * 2 + 1);
			float *p4 = PointerAt<float>(input, x * 2 + 1, y * 2 + 1);
			float *p = PointerAt<float>(*output, x, y);
			*p = (*p1 + *p2 + *p3 + *p4) / 4.0f;
		}
	}
	return output;
}

std::shared_ptr<Image> FilterHorizontalImage(
		const Image &input, const std::vector<double> &kernel)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4 ||
			kernel.size() % 2 != 1) {
		PrintWarning("[FilterHorizontalImage] Unsupported image format or kernel size.\n");
		return output;
	}
	output->PrepareImage(input.width_, input.height_, 1, 4);

	const int half_kernel_size = (int)(floor((double)kernel.size() / 2.0));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float* po = PointerAt<float>(*output, x, y, 0);
			double temp = 0;
			for (int i = -half_kernel_size; i <= half_kernel_size; i++) {
				int x_shift = x + i;
				if (x_shift < 0)
					x_shift = 0;
				if (x_shift > input.width_ - 1)
					x_shift = input.width_ - 1;
				float* pi = PointerAt<float>(input, x_shift, y, 0);
				temp += (*pi * (float)kernel[i + half_kernel_size]);
			}
			*po = (float)temp;
		}
	}
	return output;
}

std::shared_ptr<Image> FilterImage(const Image &input, Image::FilterType type)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[FilterImage] Unsupported image format.\n");
		return output;
	}

	switch (type) {
	case Image::FilterType::Gaussian3:
		output = FilterImage(input, Gaussian3, Gaussian3);
		break;
	case Image::FilterType::Gaussian5:
		output = FilterImage(input, Gaussian5, Gaussian5);
		break;
	case Image::FilterType::Gaussian7:
		output = FilterImage(input, Gaussian7, Gaussian7);
		break;
	case Image::FilterType::Sobel3Dx:
		output = FilterImage(input, Sobel31, Sobel32);
		break;
	case Image::FilterType::Sobel3Dy:
		output = FilterImage(input, Sobel32, Sobel31);
		break;
	default:
		PrintWarning("[FilterImage] Unsupported filter type.\n");
		break;
	}
	return output;
}

ImagePyramid FilterImagePyramid(const ImagePyramid &input,
		Image::FilterType type)
{
	std::vector<std::shared_ptr<Image>> output;
	for (size_t i = 0; i < input.size(); i++) {
		auto layer_filtered = FilterImage(*input[i], type);
		output.push_back(layer_filtered);
	}
	return output;
}

std::shared_ptr<Image> FilterImage(const Image &input,
		const std::vector<double> &dx, const std::vector<double> &dy)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[FilterImage] Unsupported image format.\n");
		return output;
	}

	auto temp1 = FilterHorizontalImage(input, dx);
	auto temp2 = FlipImage(*temp1);
	auto temp3 = FilterHorizontalImage(*temp2, dy);
	auto temp4 = FlipImage(*temp3);
	return temp4;
}

std::shared_ptr<Image> FlipImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintWarning("[FilpImage] Unsupported image format.\n");
		return output;
	}
	output->PrepareImage(input.height_, input.width_, 1, 4);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float* pi = PointerAt<float>(input, x, y, 0);
			float* po = PointerAt<float>(*output, y, x, 0);
			*po = *pi;
		}
	}
	return output;
}

}	// namespace three
