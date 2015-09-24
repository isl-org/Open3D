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

#include "DrawGeometry.h"

namespace three{

bool DrawGeometry(
		std::shared_ptr<const Geometry> geometry_ptr,
		const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/)
{
	Visualizer visualizer;
	if (visualizer.CreateWindow(window_name, width, height, left, top) == 
			false) {
		PrintWarning("[DrawGeometry] Failed creating OpenGL window.\n");
		return false;
	}
	if (visualizer.AddGeometry(geometry_ptr) == false) {
		PrintWarning("[DrawGeometry] Failed adding geometry.\n");
		return false;
	}
	visualizer.Run();
	return true;
}

bool DrawGeometryWithCallback(
		std::shared_ptr<const Geometry> geometry_ptr,
		std::function<bool(Visualizer &)> callback_func,
		const std::string window_name/* = "Open3DV"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/)
{
	Visualizer visualizer;
	if (visualizer.CreateWindow(window_name, width, height, left, top) == 
			false) {
		PrintWarning("[DrawGeometry] Failed creating OpenGL window.\n");
		return false;
	}
	if (visualizer.AddGeometry(geometry_ptr) == false) {
		PrintWarning("[DrawGeometry] Failed adding geometry.\n");
		return false;
	}
	while (visualizer.PollEvents()) {
		if (callback_func(visualizer)) {
			visualizer.UpdateGeometry();
		}

		// Set render flag as dirty anyways, because when we use callback
		// functions, we assume something has been changed in the callback and
		// the redraw event should be triggered.
		visualizer.UpdateRender();
	}
	visualizer.Run();
	return true;
}

}	// namespace three
