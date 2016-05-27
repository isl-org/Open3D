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

#include <Core/Geometry/Image.h>

namespace three {

class FloatImage : public Image
{
public:
	FloatImage() {}
	~FloatImage() override {}

public:
	bool HasData() const override {
		return num_of_channels_ == 1 && bytes_per_channel_ == 4 && 
				Image::HasData();
	}

public:
	void PrepareImage(int width, int height) {
		Image::PrepareImage(width, height, 1, 4);
	}
	std::pair<bool, double> ValueAt(double u, double v);

protected:
	float ValueAtUnsafe(int u, int v) {
		return *((float *)(data_.data() + (u + v * width_) * 4));
	}
};

/// Factory function to create a FloatImage from an image (ImageFactory.cpp)
std::shared_ptr<FloatImage> CreateFloatImageFromImage(const Image &image);

}	// namespace three
