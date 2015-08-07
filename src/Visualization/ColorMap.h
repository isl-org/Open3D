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

#pragma once

#include <Eigen/Core>

namespace three {

enum ColorMapOption {
	COLORMAP_GRAY = 0,
	COLORMAP_JET = 1,
	COLORMAP_SUMMER = 2,
	COLORMAP_WINTER = 3,
};

class ColorMap  {
public:
	ColorMap();
	~ColorMap();

public:
	/// Function to get a color from a value in [0..1]
	virtual Eigen::Vector3d GetColor(double value) = 0;

protected:
	double Interpolate(double value, 
			double y0, double x0, double y1, double x1)
	{
		return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
	}
};

class ColorMapGray : public ColorMap {
public:
	virtual Eigen::Vector3d GetColor(double value);
};

/// See Matlab's Jet colormap
class ColorMapJet : public ColorMap {
public:
	virtual Eigen::Vector3d GetColor(double value);

protected:
	double JetBase(double value) {
		if (value <= -0.75) { 
			return 0.0;
		} else if (value <= -0.25) {
			return Interpolate(value, 0.0, -0.75, 1.0, -0.25);
		} else if (value <= 0.25) {
			return 1.0;
		} else if (value <= 0.75) {
			return Interpolate(value, 1.0, 0.25, 0.0, 0.75);
		} else {
			return 0.0;
		}
	}
};

/// See Matlab's Summer colormap
class ColorMapSummer : public ColorMap {
public:
	virtual Eigen::Vector3d GetColor(double value);
};

/// See Matlab's Winter colormap
class ColorMapWinter : public ColorMap {
public:
	virtual Eigen::Vector3d GetColor(double value);
};

}	// namespace three
