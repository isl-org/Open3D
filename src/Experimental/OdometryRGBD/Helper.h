// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Jaesik Park <syncle@gmail.com>
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


namespace {

//	enum FILTER_TYPE {
//		FILTER_GAUSSIAN_3,
//		FILTER_GAUSSIAN_5,
//		FILTER_GAUSSIAN_7,
//		FILTER_SOBEL_3
//	};
//
//const std::vector<double> Gaussian = 
//	{ 0.0113, 0.0838, 0.0113,
//	0.0838, 0.6193, 0.0838,
//	0.0113, 0.0838, 0.0113 };
////// Gaussian filter coefficients
////// same as how the gaussian kernel is obtained 
////const std::vector<double> Gaussian3 =
////{ 0.0571    0.1248    0.0571
////	0.1248    0.2725    0.1248
////	0.0571    0.1248    0.0571 };
////const std::vector<double> Gaussian5 =
////{ 0.0050    0.0173    0.0262    0.0173    0.0050
////	0.0173    0.0598    0.0903    0.0598    0.0173
////	0.0262    0.0903    0.1366    0.0903    0.0262
////	0.0173    0.0598    0.0903    0.0598    0.0173
////	0.0050    0.0173    0.0262    0.0173    0.0050 };
////const std::vector<double> Gaussian7 =
////{ 0.0008    0.0030    0.0065    0.0084    0.0065    0.0030    0.0008
////	0.0030    0.0108    0.0232    0.0299    0.0232    0.0108    0.0030
////	0.0065    0.0232    0.0498    0.0643    0.0498    0.0232    0.0065
////	0.0084    0.0299    0.0643    0.0830    0.0643    0.0299    0.0084
////	0.0065    0.0232    0.0498    0.0643    0.0498    0.0232    0.0065
////	0.0030    0.0108    0.0232    0.0299    0.0232    0.0108    0.0030
////	0.0008    0.0030    0.0065    0.0084    0.0065    0.0030    0.0008 };
//
//// Sobel filter coefficients
//const double divfac = 8.0f; // damping factor
//const std::vector<double> Sobel_dx =
//{ -1.0f / divfac, 0.0f / divfac, 1.0f / divfac,
//	-2.0f / divfac, 0.0f / divfac, 2.0f / divfac,
//	-1.0f / divfac, 0.0f / divfac, 1.0f / divfac };
//const std::vector<double> Sobel_dy =
//{ -1.0f / divfac, -2.0f / divfac, -1.0f / divfac,
//	0.0f / divfac, 0.0f / divfac, 0.0f / divfac,
//	1.0f / divfac, 2.0f / divfac, 1.0f / divfac };


} // unnamed namespace

namespace three {






//// some helper functions
//void ConvertDepthToFloatImage(const Image &depth, Image &depth_f,
//	double depth_scale/* = 1000.0*/, double depth_trunc/* = 3.0*/);




//void BuildingPyramidImage(
//	const Image &image,
//	std::vector<std::shared_ptr<const Image>> &pyramidImage,
//	std::vector<std::shared_ptr<const Image>> &pyramidImageGradx,
//	std::vector<std::shared_ptr<const Image>> &pyramidImageGrady,
//	size_t levelCount);

}	// namespace three
