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

#include <memory>
#include <Core/Utility/Console.h>
#include <Core/Geometry/Image.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>

namespace three {

const float Gaussian[9] = { 0.0113, 0.0838, 0.0113,
		0.0838, 0.6193, 0.0838,
		0.0113, 0.0838, 0.0113 };
const float divfac = 8.0f; // damping factor
const float Sobel_dx[9] = { -1.0f / divfac, 0.0f / divfac, 1.0f / divfac,
		-2.0f / divfac, 0.0f / divfac, 2.0f / divfac,
		-1.0f / divfac, 0.0f / divfac, 1.0f / divfac };
const float Sobel_dy[9] = { -1.0f / divfac, -2.0f / divfac, -1.0f / divfac,
		0.0f / divfac, 0.0f / divfac, 0.0f / divfac,
		1.0f / divfac, 2.0f / divfac, 1.0f / divfac };

void ConvertDepthToFloatImage(const Image &depth, Image &depth_f,
  double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/);

void PreprocessDepth(const Image &depth);

// 3x3 filtering
// assumes single channel float type image
std::shared_ptr<Image> FilteringImage(const Image &input, const float *kernel);

// 2x image downsampling
// assumes float type image
// simple 2x2 averaging
// assumes 2x powered image width and height
// need to double check how we are going to handle invalid depth
std::shared_ptr<Image> DownsamplingImage(const Image &input);

void BuildingPyramidImage(const Image& image,
  std::vector<std::shared_ptr<const Image>>& pyramidImage,
  size_t levelCount);

void BuildingPyramidImage(
  const Image &image,
  std::vector<std::shared_ptr<const Image>> &pyramidImage,
  std::vector<std::shared_ptr<const Image>> &pyramidImageGradx,
  std::vector<std::shared_ptr<const Image>> &pyramidImageGrady,
  size_t levelCount);


}	// namespace three
