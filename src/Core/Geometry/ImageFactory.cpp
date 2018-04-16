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

#include <Core/Camera/PinholeCameraIntrinsic.h>
#include <IO/ClassIO/ImageIO.h>

namespace three{

std::shared_ptr<Image> CreateImageFromFile(const std::string &filename)
{
	auto image = std::make_shared<Image>();
	ReadImage(filename, *image);
	return image;
}

std::shared_ptr<Image> CreateDepthToCameraDistanceMultiplierFloatImage(
		const PinholeCameraIntrinsic &intrinsic)
{
	auto fimage = std::make_shared<Image>();
	fimage->PrepareImage(intrinsic.width_, intrinsic.height_, 1, 4);
	float ffl_inv[2] = {
			1.0f / (float)intrinsic.GetFocalLength().first,
			1.0f / (float)intrinsic.GetFocalLength().second,
	};
	float fpp[2] = {
			(float)intrinsic.GetPrincipalPoint().first,
			(float)intrinsic.GetPrincipalPoint().second,
	};
	std::vector<float> xx(intrinsic.width_);
	std::vector<float> yy(intrinsic.height_);
	for (int j = 0; j < intrinsic.width_; j++) {
		xx[j] = (j - fpp[0]) * ffl_inv[0];
	}
	for (int i = 0; i < intrinsic.height_; i++) {
		yy[i] = (i - fpp[1]) * ffl_inv[1];
	}
	for (int i = 0; i < intrinsic.height_; i++) {
		float *fp = (float *)(fimage->data_.data() +
				i * fimage->BytesPerLine());
		for (int j = 0; j < intrinsic.width_; j++, fp++) {
			*fp = sqrtf(xx[j] * xx[j] + yy[i] * yy[i] + 1.0f);
		}
	}
	return fimage;
}

std::shared_ptr<Image> CreateFloatImageFromImage(const Image &image,
		Image::ColorToIntensityConversionType type/* = WEIGHTED*/)
{
	auto fimage = std::make_shared<Image>();
	if (image.IsEmpty()) {
		return fimage;
	}
	fimage->PrepareImage(image.width_, image.height_, 1, 4);
	for (int i = 0; i < image.height_ * image.width_; i++) {
		float *p = (float *)(fimage->data_.data() + i * 4);
		const uint8_t *pi = image.data_.data() +
				i * image.num_of_channels_ * image.bytes_per_channel_;
		if (image.num_of_channels_ == 1) {
			// grayscale image
			if (image.bytes_per_channel_ == 1) {
				*p = (float)(*pi) / 255.0f;
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				*p = (float)(*pi16);
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				*p = *pf;
			}
		} else if (image.num_of_channels_ == 3) {
			if (image.bytes_per_channel_ == 1) {
				if (type == Image::ColorToIntensityConversionType::Equal) {
					*p = ((float)(pi[0]) + (float)(pi[1]) + (float)(pi[2])) /
						3.0f / 255.0f;
				} else if (type ==
						Image::ColorToIntensityConversionType::Weighted) {
					*p = (0.2990f * (float)(pi[0]) + 0.5870f * (float)(pi[1]) +
							0.1140f * (float)(pi[2])) / 255.0f;
				}
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				if (type == Image::ColorToIntensityConversionType::Equal) {
					*p = ((float)(pi16[0]) + (float)(pi16[1]) +
							(float)(pi16[2])) / 3.0f;
				} else if (type ==
						Image::ColorToIntensityConversionType::Weighted) {
					*p = (0.2990f * (float)(pi16[0]) +
							0.5870f * (float)(pi16[1]) +
							0.1140f * (float)(pi16[2]));
				}
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				if (type == Image::ColorToIntensityConversionType::Equal) {
					*p = (pf[0] + pf[1] + pf[2]) / 3.0f;
				} else if (type ==
						Image::ColorToIntensityConversionType::Weighted) {
					*p = (0.2990f * pf[0] + 0.5870f * pf[1] + 0.1140f * pf[2]);
				}
			}
		}
	}
	return fimage;
}

template <typename T>
std::shared_ptr<Image> CreateImageFromFloatImage(const Image &input)
{
	auto output = std::make_shared<Image>();
	if (input.num_of_channels_ != 1 ||
			input.bytes_per_channel_ != 4) {
		PrintDebug("[CreateImageFromFloatImage] Unsupported image format.\n");
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

template std::shared_ptr<Image> CreateImageFromFloatImage<uint8_t>(
		const Image &input);
template std::shared_ptr<Image> CreateImageFromFloatImage<uint16_t>(
		const Image &input);

ImagePyramid CreateImagePyramid(
		const Image &input, size_t num_of_levels,
		bool with_gaussian_filter /*= true*/)
{
	std::vector<std::shared_ptr<Image>> pyramid_image;
	pyramid_image.clear();
	if ((input.num_of_channels_ != 1) || (input.bytes_per_channel_ != 4)) {
		PrintWarning("[CreateImagePyramid] Unsupported image format.\n");
		return pyramid_image;
	}

	for (int i = 0; i < num_of_levels; i++) {
		if (i == 0) {
			std::shared_ptr<Image> input_copy_ptr = std::make_shared<Image>();
			*input_copy_ptr = input;
			pyramid_image.push_back(input_copy_ptr);
		}
		else {
			if (with_gaussian_filter) {
				// https://en.wikipedia.org/wiki/Pyramid_(image_processing)
				auto level_b = FilterImage(*pyramid_image[i - 1],
						Image::FilterType::Gaussian3);
				auto level_bd = DownsampleImage(*level_b);
				pyramid_image.push_back(level_bd);
			}
			else {
				auto level_d = DownsampleImage(*pyramid_image[i - 1]);
				pyramid_image.push_back(level_d);
			}
		}
	}
	return pyramid_image;
}

}	// namespace three
