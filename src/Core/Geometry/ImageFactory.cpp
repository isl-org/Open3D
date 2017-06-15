// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Qianyi Zhou <Qianyi.Zhou@gmail.com>
//                    Jaesik Park <syncle@gmail.com>
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
#include "RGBDImage.h"

#include <IO/ClassIO/ImageIO.h>

namespace three{

std::shared_ptr<Image> CreateImageFromFile(const std::string &filename)
{
	auto image = std::make_shared<Image>();
	ReadImage(filename, *image);
	return image;
}

std::shared_ptr<Image> CreateFloatImageFromImage(
		const Image &image, AverageType average_type/* = WEIGHTED*/)
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
				*p = (float)(*pi16);
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
					*p = (0.2990f * (float)(pi[0]) + 0.5870f * (float)(pi[1]) +
							0.1140f * (float)(pi[2])) / 255.0f;
				}				
			} else if (image.bytes_per_channel_ == 2) {
				const uint16_t *pi16 = (const uint16_t *)pi;
				if (average_type == EQUAL) {
					*p = ((float)(pi16[0]) + (float)(pi16[1]) + 
							(float)(pi16[2])) / 3.0f;
				} else if (average_type == WEIGHTED) {
					*p = (0.2990f * (float)(pi16[0]) + 
							0.5870f * (float)(pi16[1]) + 
							0.1140f * (float)(pi16[2]));
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

ImagePyramid CreateImagePyramid(
		const Image& input, size_t num_of_levels,
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
				auto level_b = FilterImage(*pyramid_image[i - 1], FILTER_GAUSSIAN_3);
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

std::shared_ptr<RGBDImage> CreateRGBDImageFromColorAndDepth(
		const Image& color, const Image& depth, 
		const double& depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/) {
	std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
	if (color.height_ != depth.height_ || color.width_ != depth.width_) {
		PrintWarning("[CreateRGBDImageFromColorAndDepth] Unsupported image format.\n");
		return rgbd_image;
	}
	auto color_f = CreateFloatImageFromImage(color);
	auto depth_f = ConvertDepthToFloatImage(depth, depth_scale, depth_trunc);
	rgbd_image->color_ = *color_f;
	rgbd_image->depth_ = *depth_f;
	return rgbd_image;
}

std::shared_ptr<RGBDImage> CreateRGBDImageFromTUMFormat(
		const Image& color, const Image& depth) {
	return CreateRGBDImageFromColorAndDepth(color, depth, 5000.0);
}

std::shared_ptr<RGBDImage> CreateRGBDImageFromSUNFormat(
		const Image& color, const Image& depth) {
	std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
	if (color.height_ != depth.height_ || color.width_ != depth.width_) {
		PrintWarning("[CreateRGBDImageFromSUNFormat] Unsupported image format.\n");
		return rgbd_image;
	}
	for (int v = 0; v < depth.height_; v += 1) {
		for (int u = 0; u < depth.width_; u += 1) {
			unsigned short & d = *PointerAt<unsigned short>(depth, u, v);
			d = (d >> 3) | (d << 13);
		}
	}
	auto color_f = CreateFloatImageFromImage(color);
	// SUN depth map has long range depth. We set depth_trunc as 7.0
	auto depth_f = ConvertDepthToFloatImage(depth, 1000, 7.0);
	rgbd_image->color_ = *color_f;
	rgbd_image->depth_ = *depth_f;
	return rgbd_image;
}

std::shared_ptr<RGBDImage> CreateRGBDImageFromNYUFormat(
		const Image& color, const Image& depth) {
	std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
	if (color.height_ != depth.height_ || color.width_ != depth.width_) {
		PrintWarning("[CreateRGBDImageFromNYUFormat] Unsupported image format.\n");
		return rgbd_image;
	}
	for (int v = 0; v < depth.height_; v += 1) {
		for (int u = 0; u < depth.width_; u += 1) {
			unsigned short * d = PointerAt<unsigned short>(depth, u, v);
			unsigned char * p = (unsigned char *)d;
			unsigned char x = *p;
			*p = *(p + 1);
			*(p + 1) = x;
			double xx = 351.3 / (1092.5 - *d);
			if (xx <= 0.0) {
				*d = 0;
			} else {
				*d = (unsigned short)(floor(xx * 1000 + 0.5));
			}
		}
	}
	auto color_f = CreateFloatImageFromImage(color);
	// NYU depth map has long range depth. We set depth_trunc as 7.0
	auto depth_f = ConvertDepthToFloatImage(depth, 1000, 7.0);
	rgbd_image->color_ = *color_f;
	rgbd_image->depth_ = *depth_f;
	return rgbd_image;
}

}	// namespace three
