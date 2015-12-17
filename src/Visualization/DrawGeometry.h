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

#include <string>
#include <memory>
#include <functional>
#include <Core/Geometry.h>

#include "Visualizer.h"
#include "VisualizerWithAnimation.h"

namespace three {

/// The convenient function of drawing something
/// This function is a wrapper that calls the core functions of Visualizer.
/// This function MUST be called from the main thread. It blocks the main thread
/// until the window is closed.
bool DrawGeometry(
		std::shared_ptr<const Geometry> geometry_ptr,
		const std::string &window_name = "Open3D", 
		const int width = 640, const int height = 480,
		const int left = 50, const int top = 50);

bool DrawGeometryWithAnimation(
		std::shared_ptr<const Geometry> geometry_ptr,
		const std::string &window_name = "Open3D", 
		const int width = 640, const int height = 480,
		const int left = 50, const int top = 50,
		const std::string &json_filename = "");

bool DrawGeometryWithCallback(
		std::shared_ptr<const Geometry> geometry_ptr,
		std::function<bool(Visualizer &)> callback_func,
		const std::string &window_name = "Open3D", 
		const int width = 640, const int height = 480,
		const int left = 50, const int top = 50);

}	// namespace three
