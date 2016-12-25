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

#include "SelectionPolygon.h"

namespace three{

void SelectionPolygon::Clear()
{
	polygon_.clear();
	is_closed_ = false;
	polygon_interior_mask_.Clear();
}

bool SelectionPolygon::IsEmpty() const
{
	// A valid polygon, either close or open, should have at least 2 vertices.
	return polygon_.size() <= 1;
}

void SelectionPolygon::FillPolygon(int width, int height)
{
	// Standard scan conversion code. See reference:
	// http://alienryderflex.com/polygon_fill/
	
	if (IsEmpty()) {
		return;
	}
	is_closed_ = true;
	polygon_interior_mask_.PrepareImage(width, height, 1, 1);
	std::fill(polygon_interior_mask_.data_.begin(),
			polygon_interior_mask_.data_.end(), 0);
	std::vector<int> nodes;
	for (int y = 0; y < height; y++) {
		nodes.clear();
		for (size_t i = 0; i < polygon_.size(); i++) {
			size_t j = (i + 1) % polygon_.size();
			if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
					(polygon_[j](1) < y && polygon_[i](1) >= y)) {
				nodes.push_back((int)(polygon_[i](0) + (y - polygon_[i](1)) /
						(polygon_[j](1) - polygon_[i](1)) * (polygon_[j](0) -
						polygon_[i](0)) + 0.5));
			}
		}
		std::sort(nodes.begin(), nodes.end());
		for (size_t i = 0; i < nodes.size(); i+= 2) {
			if (nodes[i] >= width) {
				break;
			}
			if (nodes[i + 1] > 0) {
				if (nodes[i] < 0) nodes[i] = 0;
				if (nodes[i + 1] > width) nodes[i + 1] = width;
				for (int x = nodes[i]; x < nodes[i + 1]; x++) {
					polygon_interior_mask_.data_[x + y * width] = 1;
				}
			}
		}
	}
}

}	// namespace three
