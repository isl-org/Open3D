// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncel@gmail.com>
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

class RGBDImage : public Geometry2D
{
public:
	RGBDImage() : Geometry2D(GEOMETRY_IMAGE) {};
	~RGBDImage() override {};

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector2d GetMinBound() const override;
	Eigen::Vector2d GetMaxBound() const override;

public:
	Image depth;
	Image color;
};

/// Factory function to create an RGBD Image from color and depth Images
std::shared_ptr<RGBDImage> CreateRGBDImageFromColorAndDepth(
		const Image& color, const Image& depth, 
		const double& depth_scale = 1000.0, double depth_trunc = 3.0);

std::shared_ptr<RGBDImage> CreateRGBDImageFromTUMFormat(
		const Image& color, const Image& depth);

}	// namespace three
