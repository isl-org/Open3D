// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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

#include "ColorMap.h"

namespace three{

namespace {

static std::shared_ptr<ColorMap> global_colormap_ptr(new ColorMapJet);

}	// unnamed namespace

Eigen::Vector3d ColorMapGray::GetColor(double value) const
{
	return Eigen::Vector3d(value, value, value);
}

Eigen::Vector3d ColorMapJet::GetColor(double value) const
{
	return Eigen::Vector3d(
			JetBase(value * 2.0 - 1.5),		// red
			JetBase(value * 2.0 - 1.0),		// green
			JetBase(value * 2.0 - 0.5));	// blue
}

Eigen::Vector3d ColorMapSummer::GetColor(double value) const
{
	return Eigen::Vector3d(
			Interpolate(value, 0.0, 0.0, 1.0, 1.0),
			Interpolate(value, 0.5, 0.0, 1.0, 1.0),
			0.4);
}

Eigen::Vector3d ColorMapWinter::GetColor(double value) const
{
	return Eigen::Vector3d(
			0.0,
			Interpolate(value, 0.0, 0.0, 1.0, 1.0),
			Interpolate(value, 1.0, 0.0, 0.5, 1.0));
}

std::shared_ptr<const ColorMap> GetGlobalColorMap()
{
	return global_colormap_ptr;
}

void SetGlobalColorMap(ColorMap::ColorMapOption option)
{
	switch (option) {
	case ColorMap::COLORMAP_GRAY:
		global_colormap_ptr.reset(new ColorMapGray);
		break;
	case ColorMap::COLORMAP_SUMMER:
		global_colormap_ptr.reset(new ColorMapSummer);
		break;
	case ColorMap::COLORMAP_WINTER:
		global_colormap_ptr.reset(new ColorMapWinter);
		break;
	case ColorMap::COLORMAP_JET:
	default:
		global_colormap_ptr.reset(new ColorMapJet);
		break;
	}
}

}	// namespace three
