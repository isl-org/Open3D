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

#include <IO/ClassIO/ImageIO.h>

namespace three{

std::shared_ptr<Image> CreateImageFromFile(const std::string &filename)
{
	auto image = std::make_shared<Image>();
	ReadImage(filename, *image);
	return image;
}

std::shared_ptr<Image> CreateFloatImageFromImage(const Image &image, AverageType average_type/* = WEIGHTED*/)
{
	auto fimage = std::make_shared<Image>();
	if (image.IsEmpty() || 
		((average_type != EQUAL) && (average_type != WEIGHTED))) {
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
				*p = (float)(*pi16) / 65535.0f;
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				*p = *pf;
			}
		} else if (image.num_of_channels_ == 3) {
			if (image.bytes_per_channel_ == 1) {
				if (average_type == EQUAL) {
					*p = ((float)(pi[0]) + (float)(pi[1]) + (float)(pi[2])) /
						3.0f / 255.0f;
				} else if (average_type == WEIGHTED) {
					*p = (0.2990f * (float)(pi[0]) + 0.5870f * (float)(pi[1]) + 0.1140f * (float)(pi[2]))
						/ 255.0f;
				}				
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				if (average_type == EQUAL) {
					*p = ((float)(pi16[0]) + (float)(pi16[1]) + (float)(pi16[2])) /
						3.0f / 65535.0f;
				} else if (average_type == WEIGHTED) {
					*p = (0.2990f * (float)(pi16[0]) + 0.5870f * (float)(pi16[1]) + 0.1140f * (float)(pi16[2]))
						/ 65535.0f;
				}
			} else if (image.bytes_per_channel_ == 4) {
				const float *pf = (const float *)pi;
				if (average_type == EQUAL) {
					*p = (pf[0] + pf[1] + pf[2]) / 3.0f;
				} else if (average_type == WEIGHTED) {
					*p = (0.2990f * pf[0] + 0.5870f * pf[1] + 0.1140f * pf[2]);
				}
			}
		}
	}
	return fimage;
}

}	// namespace three
