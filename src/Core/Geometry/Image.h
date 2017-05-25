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

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>

#include <Core/Geometry/Geometry2D.h>

namespace three {

class Image : public Geometry2D
{
public:
	Image() : Geometry2D(GEOMETRY_IMAGE) {};
	~Image() override {};

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector2d GetMinBound() const override;
	Eigen::Vector2d GetMaxBound() const override;

public:
	virtual bool HasData() const {
		return width_ > 0 && height_ > 0 && 
				data_.size() == height_ * BytesPerLine();
	}

	void PrepareImage(int width, int height, int num_of_channels, 
			int bytes_per_channel) {
		width_ = width;
		height_ = height;
		num_of_channels_ = num_of_channels;
		bytes_per_channel_ = bytes_per_channel;
		AllocateDataBuffer();
	}

	int BytesPerLine() const {
		return width_ * num_of_channels_ * bytes_per_channel_;
	}

	float FloatValueAtUnsafe(int u, int v) {
		return *((float *)(data_.data() + (u + v * width_) * bytes_per_channel_));
	}

	std::pair<bool, double> FloatValueAt(double u, double v);
		
protected:
	void AllocateDataBuffer() {
		data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
	}
	
public:
	int width_ = 0;
	int height_ = 0;
	int num_of_channels_ = 0;
	int bytes_per_channel_ = 0;
	std::vector<uint8_t> data_;
};

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<Image> CreateImageFromFile(const std::string &filename);

/// Return an gray scaled float type image.
std::shared_ptr<Image> CreateFloatImageFromImage(const Image &image);


}	// namespace three
