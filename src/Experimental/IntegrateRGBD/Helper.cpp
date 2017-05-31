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

#include "Helper.h"

namespace three{

std::shared_ptr<Image> CreateDepthToCameraDistanceConversionImage(
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

}	// namespace three
