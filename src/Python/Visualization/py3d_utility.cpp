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

#include "py3d_visualization.h"

#include <Visualization/Visualization.h>
using namespace three;

void pybind_utility_methods(py::module &m)
{
	m.def("DrawGeometries", &DrawGeometries,
			"Function to draw a list of Geometry objects",
			"geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
			"height"_a = 1080, "left"_a = 50, "top"_a = 50);
	m.def("DrawGeometriesWithCustomAnimation",
			&DrawGeometriesWithCustomAnimation,
			"Function to draw a list of Geometry objects",
			"geometry_list"_a, "window_name"_a = "Open3D", "width"_a = 1920,
			"height"_a = 1080, "left"_a = 50, "top"_a = 50,
			"optional_view_trajectory_json_file"_a = "");	
}
