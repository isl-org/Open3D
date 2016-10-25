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
#include "FloatImage.h"

#include <IO/ClassIO/ImageIO.h>

namespace three{

std::shared_ptr<Image> CreateImageFromFile(const std::string &filename)
{
	auto image = std::make_shared<Image>();
	ReadImage(filename, *image);
	return image;
}

std::shared_ptr<FloatImage> CreateFloatImageFromImage(const Image &image)
{
	auto fimage = std::make_shared<FloatImage>();
	if (image.IsEmpty()) {
		return fimage;
	}
	fimage->PrepareImage(image.width_, image.height_);
	for (int i = 0; i < image.height_ * image.width_; i++) {
		float *p = (float *)(fimage->data_.data() + i * 4);
		const unsigned char *pi = image.data_.data() + 
				i * image.num_of_channels_ * image.bytes_per_channel_;
		if (image.num_of_channels_ == 1) {
			// grayscale image
			if (image.bytes_per_channel_ == 1) {
				*p = (float)(*pi) / 255.0f;
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				*p = (float)(*pi16) / 65535.0f;
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				*p = *pf;
			}
		} else if (image.num_of_channels_ == 3) {
			if (image.bytes_per_channel_ == 1) {
				*p = ((float)(pi[0]) + (float)(pi[1]) + (float)(pi[2])) / 
						3.0f / 255.0f;
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				*p = ((float)(pi16[0]) + (float)(pi16[1]) + (float)(pi16[2])) /
						3.0f / 65535.0f;
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				*p = (pf[0] + pf[1] + pf[2]) / 3.0f;
			}
		}
	}
	return fimage;
}

std::shared_ptr<FloatImage> CreateCameraDistanceFloatImageFromDepthImage(
		const Image &depth, const PinholeCameraIntrinsic &intrinsic,
		double depth_scale/* = 1000.0*/)
{
	auto fimage = std::make_shared<FloatImage>();
	if (depth.IsEmpty() || depth.num_of_channels_ != 1 ||
			depth.bytes_per_channel_ != 2) {
		return fimage;
	}
	fimage->PrepareImage(depth.width_, depth.height_);
	// Compute camera_distance = sqrt(x * x + y * y + z * z)
	// The following code has been optimized for speed
	float ffl_inv[2] = {
			1.0f / (float)intrinsic.GetFocalLength().first,
			1.0f / (float)intrinsic.GetFocalLength().second,
	};
	float fpp[2] = {
			(float)intrinsic.GetPrincipalPoint().first,
			(float)intrinsic.GetPrincipalPoint().second,
	};
	float fd_inv = 1.0f / (float)depth_scale;
	std::vector<float> xx(depth.width_);
	std::vector<float> yy(depth.height_);
	for (int i = 0; i < depth.height_; i++) {
		yy[i] = (i - fpp[1]) * ffl_inv[1];
	}
	for (int j = 0; j < depth.width_; j++) {
		xx[j] = (j - fpp[0]) * ffl_inv[0];
	}
	for (int i = 0; i < depth.height_; i++) {
		uint16_t *p = (uint16_t *)(depth.data_.data() + 
				i * depth.BytesPerLine());
		float *fp = (float *)(fimage->data_.data() +
				i * fimage->BytesPerLine());
		for (int j = 0; j < depth.width_; j++, p++, fp++) {
			if (*p == 0) {
				*fp = 0.0f;
			} else {
				*fp = sqrtf(xx[j] * xx[j] + yy[i] * yy[i] + 1.0f) * (*p) *
						fd_inv;
			}
		}
	}
	return fimage;
}

}	// namespace three
