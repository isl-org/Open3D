// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

namespace three{

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

std::pair<bool, double> Image::FloatValueAt(double u, double v)
{
	if ((num_of_channels_ != 1) ||
		(bytes_per_channel_ != 4) ||
		(u < 0.0 || u >(double)(width_ - 1) ||
		v < 0.0 || v >(double)(height_ - 1))) {
		return std::make_pair(false, 0.0);
	}
	int ui = std::max(std::min((int)u, width_ - 2), 0);
	int vi = std::max(std::min((int)v, height_ - 2), 0);
	double pu = u - ui;
	double pv = v - vi;
	float value[4] = {
		FloatValueAtUnsafe(ui, vi),
		FloatValueAtUnsafe(ui, vi + 1),
		FloatValueAtUnsafe(ui + 1, vi),
		FloatValueAtUnsafe(ui + 1, vi + 1)
	};
	return std::make_pair(true,
		(value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
		(value[2] * (1 - pv) + value[3] * pv) * pu);
}

std::shared_ptr<Image> ConvertDepthToFloatImage(const Image &depth,
		double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/) 
{
	if (depth.num_of_channels_ != 1 ||
		depth.bytes_per_channel_ != 2) {
		PrintDebug("[ConvertDepthToFloatImage] Unsupported image format.\n");
		return std::make_shared<Image>();
	}
	auto output = CreateFloatImageFromImage(depth);
	LinearTransformImage(*output, 1 / depth_scale, 0.0, 0.0, depth_trunc);
	return output;
}

void LinearTransformImage(Image &input, double scale, 
		double offset, double min, double max) 
{
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float *p = PointerAt<float>(input, x, y);
			(*p) = (float)(scale * (*p) + offset);
			if (*p > max)
				*p = (float)max;
			if (*p < min)
				*p = (float)min;
		}
	}
}

std::vector<std::shared_ptr<Image>> CreateImagePyramid(
		const Image& input, size_t num_of_levels)
{
	std::vector<std::shared_ptr<Image>> pyramidImage;
	pyramidImage.clear(); 
	if ((input.num_of_channels_ != 1) ||
		(input.bytes_per_channel_ != 4)) {
		PrintDebug("[CreateImagePyramid] Unsupported image format.\n");
		return pyramidImage;
	}

	for (int i = 0; i < num_of_levels; i++) {
		if (i == 0) {
			std::shared_ptr<Image> input_copy_ptr = 
					std::make_shared<Image>();
			*input_copy_ptr = input;
			pyramidImage.push_back(input_copy_ptr);
		} else {
			// https://en.wikipedia.org/wiki/Pyramid_(image_processing)
			auto level_b = FilterImage(*pyramidImage[i - 1], FILTER_GAUSSIAN_3);
			auto level_bd = DownsampleImage(*level_b);
			pyramidImage.push_back(level_bd);
		}
	}
	return pyramidImage;
}

std::shared_ptr<Image> DownsampleImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
		input.bytes_per_channel_ != 4) {
		PrintDebug("[DownsampleImage] Unsupported image format.\n");
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
	if (input.num_of_channels_ != 1 || 
		input.bytes_per_channel_ != 4 ||
		kernel.size() % 2 != 1) {
		PrintDebug("[FilterHorizontalImage] Unsupported image format or kernel size.\n");
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
				temp += (*pi * kernel[i + half_kernel_size]);
			}
			*po = (float)temp;
		}
	}
	return output;
}

std::shared_ptr<Image> FilterImage(const Image &input, FilterType type)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
		input.bytes_per_channel_ != 4) {
		PrintDebug("[FilterImage] Unsupported image format.\n");
		return output;
	}

	switch (type) {
		case FILTER_GAUSSIAN_3:
			output = FilterImage(input, Gaussian3, Gaussian3);
			break;
		case FILTER_GAUSSIAN_5:
			output = FilterImage(input, Gaussian5, Gaussian5);
			break;
		case FILTER_GAUSSIAN_7:
			output = FilterImage(input, Gaussian7, Gaussian7);
			break;
		case FILTER_SOBEL_3_DX:
			output = FilterImage(input, Sobel31, Sobel32);
			break;
		case FILTER_SOBEL_3_DY:
			output = FilterImage(input, Sobel32, Sobel31);
			break;
		default:
			PrintDebug("[FilterImage] Unsupported filter type.\n");
			break;
	}
	return output;
}

std::shared_ptr<Image> FilterImage(const Image &input, 
		const std::vector<double> dx, const std::vector<double> dy)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
		input.bytes_per_channel_ != 4) {
		PrintDebug("[FilterImage] Unsupported image format.\n");
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
	if (input.num_of_channels_ != 1 || 
		input.bytes_per_channel_ != 4) {
		PrintDebug("[FilpImage] Unsupported image format.\n");
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
