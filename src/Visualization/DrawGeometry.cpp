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

#include "DrawGeometry.h"

#include "ViewControlWithAnimation.h"

namespace three{

bool DrawGeometry(
		std::shared_ptr<const Geometry> geometry_ptr,
		const std::string &window_name/* = "Open3D"*/, 
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

bool DrawGeometryWithAnimation(
		std::shared_ptr<const Geometry> geometry_ptr,
		const std::string &window_name/* = "Open3D"*/, 
		const int width/* = 640*/, const int height/* = 480*/,
		const int left/* = 50*/, const int top/* = 50*/,
		const std::string &json_filename/* = ""*/)
{
	VisualizerWithAnimation visualizer;
	if (visualizer.CreateWindow(window_name, width, height, left, top) == 
			false) {
		PrintWarning("[DrawGeometry] Failed creating OpenGL window.\n");
		return false;
	}
	if (visualizer.AddGeometry(geometry_ptr) == false) {
		PrintWarning("[DrawGeometry] Failed adding geometry.\n");
		return false;
	}
	auto &view_control = 
			(ViewControlWithAnimation &)visualizer.GetViewControl();
	if (json_filename.empty() == false) {
		if (view_control.LoadTrajectoryFromFile(json_filename) == false) {
			PrintWarning("[DrawGeometry] Failed loading json file.\n");
			return false;
		}
		visualizer.UpdateWindowTitle();
	}
	visualizer.Run();
	return true;
}

bool DrawGeometryWithCallback(
		std::shared_ptr<const Geometry> geometry_ptr,
		std::function<bool(Visualizer &)> callback_func,
		const std::string &window_name/* = "Open3D"*/, 
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
	visualizer.RegisterAnimationCallback(callback_func);
	visualizer.Run();
	return true;
}

}	// namespace three
