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

#include <Core/Geometry/Geometry2D.h>
#include <Core/Geometry/Image.h>

namespace three {

/// RGBDImage is for a pair of registered color and depth images,
/// viewed from the same view, of the same resolution.
/// If you have other format, convert it first.
class RGBDImage
{
public:
	RGBDImage() {};
	RGBDImage(const Image &color, const Image &depth) :
			color_(color), depth_(depth) {};
	~RGBDImage()
	{
		color_.Clear();
		depth_.Clear();
	};

public:
	Image color_;
	Image depth_;
};

/// Factory function to create an RGBD Image from color and depth Images
std::shared_ptr<RGBDImage> CreateRGBDImageFromColorAndDepth(
		const Image &color, const Image &depth,
		double depth_scale = 1000.0, double depth_trunc = 3.0,
		bool convert_rgb_to_intensity = true);

/// Factory function to create an RGBD Image from Redwood dataset
std::shared_ptr<RGBDImage> CreateRGBDImageFromRedwoodFormat(
		const Image &color, const Image &depth,
		bool convert_rgb_to_intensity = true);

/// Factory function to create an RGBD Image from TUM dataset
std::shared_ptr<RGBDImage> CreateRGBDImageFromTUMFormat(
		const Image &color, const Image &depth,
		bool convert_rgb_to_intensity = true);

/// Factory function to create an RGBD Image from SUN3D dataset
std::shared_ptr<RGBDImage> CreateRGBDImageFromSUNFormat(
		const Image &color, const Image &depth,
		bool convert_rgb_to_intensity = true);

/// Factory function to create an RGBD Image from NYU dataset
std::shared_ptr<RGBDImage> CreateRGBDImageFromNYUFormat(
		const Image &color, const Image &depth,
		bool convert_rgb_to_intensity = true);

/// Typedef and functions for RGBDImagePyramid
typedef std::vector<std::shared_ptr<RGBDImage>> RGBDImagePyramid;

RGBDImagePyramid FilterRGBDImagePyramid(
		const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type);

RGBDImagePyramid CreateRGBDImagePyramid(const RGBDImage &rgbd_image,
		size_t num_of_levels,
		bool with_gaussian_filter_for_color = true,
		bool with_gaussian_filter_for_depth = false);

}	// namespace three
