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

#include "BoundingBox.h"

namespace three{

BoundingBox::BoundingBox()
{
}

BoundingBox::BoundingBox(const Geometry3D &geometry)
{
	FitInGeometry(geometry);
}

BoundingBox::~BoundingBox()
{
}

void BoundingBox::Reset()
{
	min_bound_.setZero();
	max_bound_.setZero();
}

void BoundingBox::FitInGeometry(const Geometry3D &geometry)
{
	if (GetSize() == 0.0) {	// empty box
		min_bound_ = geometry.GetMinBound();
		max_bound_ = geometry.GetMaxBound();
	} else {
		auto geometry_min_bound = geometry.GetMinBound();
		auto geometry_max_bound = geometry.GetMaxBound();
		min_bound_(0) = std::min(min_bound_(0), geometry_min_bound(0));
		min_bound_(1) = std::min(min_bound_(1), geometry_min_bound(1));
		min_bound_(2) = std::min(min_bound_(2), geometry_min_bound(2));
		max_bound_(0) = std::max(max_bound_(0), geometry_max_bound(0));
		max_bound_(1) = std::max(max_bound_(1), geometry_max_bound(1));
		max_bound_(2) = std::max(max_bound_(2), geometry_max_bound(2));
	}
}

}	// namespace three
