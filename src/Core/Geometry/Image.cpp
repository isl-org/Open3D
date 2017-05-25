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

void ConvertDepthToFloatImage(const Image &depth, Image &depth_f,
	double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/)
{
	if (depth_f.IsEmpty()) {
		depth_f.PrepareImage(depth.width_, depth.height_, 1, 4);
	}
	float *p = (float *)depth_f.data_.data();
	const uint16_t *pi = (const uint16_t *)depth.data_.data();
	for (int i = 0; i < depth.height_ * depth.width_; i++, p++, pi++) {
		*p = (float)(*pi) / (float)depth_scale;
		if (*p >= depth_trunc) {
			*p = 0.0f;
		}
	}
}

std::vector<std::shared_ptr<const Image>> CreateImagePyramid(const Image& input,
	size_t num_of_levels)
{
	std::vector<std::shared_ptr<const Image>> pyramidImage;
	pyramidImage.clear(); // is this good for clearing? it might have some existing data

	for (int i = 0; i < num_of_levels; i++) {
		if (i == 0) {
			std::shared_ptr<Image> input_copy_ptr(new Image);
			*input_copy_ptr = input;
			pyramidImage.push_back(input_copy_ptr);
		}
		else {
			auto layer_b = FilterImage(*pyramidImage[i - 1], Gaussian);
			auto layer_bd = DownsampleImage(*layer_b);
			pyramidImage.push_back(layer_bd);
		}
	}
	return pyramidImage;
}

std::shared_ptr<Image> DownsampleImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
		input.bytes_per_channel_ != 4 ||
		(input.width_ % 2 != 0 || input.height_ % 2 != 0)) {
		PrintDebug("[CreatePointCloudFromDepthImage] Unsupported image format.\n");
		return output;
	}
	output->PrepareImage(input.width_ / 2, input.height_ / 2, 1, 4);

	float* inputdata = (float *)input.data_.data();
	float* outputdata = (float *)output->data_.data();
	for (int c = 0; c < output->num_of_channels_; c++) {
		int cpad = c * output->width_ * output->height_;
		for (int y = 0; y < output->height_; y++) {
			for (int x = 0; x < output->width_; x++) {
				float *p1 = inputdata + (cpad + y * 2 * output->width_ + x * 2);
				float *p2 = inputdata + (cpad + y * 2 * output->width_ + (x + 1) * 2);
				float *p3 = inputdata + (cpad + (y + 1) * 2 * output->width_ + x * 2);
				float *p4 = inputdata + (cpad + (y + 1) * 2 * output->width_ + (x + 1) * 2);
				float *p = outputdata + (cpad + y * output->width_ + x);
				*p = (*p1 + *p2 + *p3 + *p4) / 4.0f;
			}
		}
	}
}

std::shared_ptr<Image> FilterImage(const Image &input, const std::vector<double> &kernel)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintDebug("[FilterImage] Unsupported image format.\n");
		return output;
	}
	output->PrepareImage(input.width_, input.height_, 1, 4);

	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			double sum = 0.0f;
			for (int yb = 0; yb < 3; yb++) {
				int yy = y + (yb - 1);
				if (yy < 0 | yy >= input.height_)
					continue;
				for (int xb = 0; xb < 3; xb++) {
					int xx = x + (xb - 1);
					if (xx < 0 | xx >= input.width_)
						continue;
					// do we have user intuitive way to access pixel value?
					float *pi = (float *)input.data_.data() +
						(y * input.width_ + x);
					sum += *pi * kernel[yb * 3 + xb];
				}
			}
			float *po = (float *)output->data_.data() +
				(y * input.width_ + x);
			*po = sum;
		}
	}
}

std::shared_ptr<Image> FilterHorizontalImage(const Image &input, const std::vector<double> &kernel)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintDebug("[FilterHorizontalImage] Unsupported image format.\n");
		return output;
	}
	output->PrepareImage(input.width_, input.height_, 1, 4);

	// see the naming rule.
	int kernelCount = kernel.size();
	int kernelCountHalf = static_cast<int>(floor(static_cast<double>(kernel.size()) / 2.0f));
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float* po = PointerAt<float>(*output, x, y, 0);
			*po = 0;
			for (int i = -kernelCountHalf; i <= kernelCountHalf; i++) {
				int x_shift = x + i;
				if (x_shift < 0)
					x_shift = 0;
				if (x_shift >= input.width_)
					x_shift = input.width_ - 1;
				float* pi = PointerAt<float>(input, x_shift, y, 0);
				*po += *pi * kernel[i + kernelCountHalf];
			}
		}
	}
	return output;
}

std::shared_ptr<Image> FilterImage(const Image &input, FilterType type)
{
	//if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
	//	PrintDebug("[FilterImage] Unsupported image format.\n");
	//	return output;
	//}

	if (type == FILTER_GAUSSIAN_3) {
		auto output_f1 = FilterHorizontalImage(input, Gaussian);
		auto output_f2 = FilpImage(*output_f1);
		auto output_f3 = FilterHorizontalImage(*output_f2, Gaussian);
		auto output_f4 = FilpImage(*output_f3);
		return output_f4;
	}
}

std::shared_ptr<Image> FilpImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 || input.bytes_per_channel_ != 4) {
		PrintDebug("[FilpImage] Unsupported image format.\n");
		return output;
	}
	output->PrepareImage(input.height_, input.width_, 1, 4);
	for (int y = 0; y < input.height_; y++) {
		for (int x = 0; x < input.width_; x++) {
			float* pi = PointerAt<float>(input, x, y, 0);
			float* po = PointerAt<float>(*output, y, x, 0);
			*po = *pi;
		}
	}
	return output;
}

template <typename T>
std::shared_ptr<Image> TypecastImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	output->PrepareImage(input.height_, input.width_, input.num_of_channels_, sizeof(T));
	float *pi = input.data_.data();
	T *p = (T *)output.data_.data();	
	for (int i = 0; i < depth.height_ * depth.width_; i++, p++) {
		if (sizeof(T) == 1)
			*p = static_cast<T>(*((float *)pi) * 255.0f);
		if (sizeof(T) == 2)
			*p = static_cast<T>(*((float *)pi) * 65535.0f);
		pi += sizeof(input.bytes_per_channel_);
	}
}

}	// namespace three
