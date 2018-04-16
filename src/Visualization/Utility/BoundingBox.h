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

#include <Core/Geometry/Geometry3D.h>
#include <Core/Utility/Console.h>

namespace three {

/// A 3D bounding box class
/// It is a utility class for Visualization
class BoundingBox
{
public:
	BoundingBox();
	BoundingBox(const Geometry3D &geometry);
	~BoundingBox();

public:
	void Reset();
	void FitInGeometry(const Geometry3D &geometry);

public:
	Eigen::Vector3d GetCenter() const {
		return (min_bound_ + max_bound_) * 0.5;
	}

	double GetSize() const {
		return (max_bound_ - min_bound_).maxCoeff();
	}

	double GetXPercentage(double x) const {
		return (x - min_bound_(0)) / (max_bound_(0) - min_bound_(0));
	}

	double GetYPercentage(double y) const {
		return (y - min_bound_(1)) / (max_bound_(1) - min_bound_(1));
	}

	double GetZPercentage(double z) const {
		return (z - min_bound_(2)) / (max_bound_(2) - min_bound_(2));
	}

	std::string GetPrintInfo() const {
		char buffer[DEFAULT_IO_BUFFER_SIZE];
		snprintf(buffer, DEFAULT_IO_BUFFER_SIZE, "[(%.4f, %.4f, %.4f) - (%.4f, %.4f, %.4f)]",
				min_bound_(0), min_bound_(1), min_bound_(2),
				max_bound_(0), max_bound_(1), max_bound_(2));
		return std::string(buffer);
	}

public:
	Eigen::Vector3d min_bound_ = Eigen::Vector3d::Zero();
	Eigen::Vector3d max_bound_ = Eigen::Vector3d::Zero();
};

}	// namespace three
