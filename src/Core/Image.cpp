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

Image::Image()
{
	SetGeometryType(GEOMETRY_IMAGE);
}

Image::~Image()
{
}
	
bool Image::CloneFrom(const Geometry &reference)
{
	if (reference.GetGeometryType() != GetGeometryType()) {
		// always return when the types do not match
		return false;
	}
	Clear();
	const Image &image = static_cast<const Image &>(reference);
	width_ = image.width_;
	height_ = image.height_;
	num_of_channels_ = image.num_of_channels_;
	bytes_per_channel_ = image.bytes_per_channel_;
	data_ = image.data_;
	return true;
}

Eigen::Vector3d Image::GetMinBound() const
{
	return Eigen::Vector3d(0.0, 0.0, 0.0);
}

Eigen::Vector3d Image::GetMaxBound() const
{
	return Eigen::Vector3d(0.0, 0.0, 0.0);
}
	
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
	
void Image::Transform(const Eigen::Matrix4d &transformation)
{
	// Transform function does not perform on Image.
}

}	// namespace three
