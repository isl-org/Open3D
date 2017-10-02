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
#include "py3d_visualization_trampoline.h"

#include <Core/Geometry/Image.h>
#include <Visualization/Visualizer/Visualizer.h>
#include <Visualization/Visualizer/VisualizerWithKeyCallback.h>
using namespace three;

void pybind_visualizer(py::module &m)
{
	py::class_<Visualizer, PyVisualizer<>, std::shared_ptr<Visualizer>>
			visualizer(m, "Visualizer");
	py::detail::bind_default_constructor<Visualizer>(visualizer);
	visualizer
		.def("__repr__", [](const Visualizer &vis) {
			return std::string("Visualizer with name ") + vis.GetWindowName();
		})
		.def("CreateWindow", &Visualizer::CreateWindow,
				"Function to create a window and initialize GLFW",
				"window_name"_a = "Open3D", "width"_a = 1920, "height"_a = 1080,
				"left"_a = 50, "right"_a = 50)
		.def("DestroyWindow", &Visualizer::DestroyWindow,
				"Function to destroy a window")
		.def("RegisterAnimationCallback",
				&Visualizer::RegisterAnimationCallback,
				"Function to register a callback function for animation",
				"callback_func"_a)
		.def("Run", &Visualizer::Run, "Function to activate the window")
		.def("Close", &Visualizer::Close,
				"Function to notify the window to be closed")
		.def("AddGeometry", &Visualizer::AddGeometry,
				"Function to add geometry to the scene and create corresponding shaders",
				"geometry"_a)
		.def("GetViewControl", &Visualizer::GetViewControl,
				"Function to retrieve the associated ViewControl",
				py::return_value_policy::reference_internal)
		.def("GetRenderOption", &Visualizer::GetRenderOption,
				"Function to retrieve the associated RenderOption",
				py::return_value_policy::reference_internal)
		.def("CaptureScreenFloatBuffer", &Visualizer::CaptureScreenFloatBuffer,
				"Function to capture screen and store RGB in a float buffer",
				"do_render"_a = false)
		.def("CaptureScreenImage", &Visualizer::CaptureScreenImage,
				"Function to capture and save a screen image",
				"filename"_a, "do_render"_a = false)
		.def("CaptureDepthFloatBuffer", &Visualizer::CaptureDepthFloatBuffer,
				"Function to capture depth in a float buffer",
				"do_render"_a = false)
		.def("CaptureDepthImage", &Visualizer::CaptureDepthImage,
				"Function to capture and save a depth image",
				"filename"_a, "do_render"_a = false, "depth_scale"_a = 1000.0)
		.def("GetWindowName", &Visualizer::GetWindowName);

	py::class_<VisualizerWithKeyCallback,
			PyVisualizer<VisualizerWithKeyCallback>,
			std::shared_ptr<VisualizerWithKeyCallback>>
			visualizer_key(m, "VisualizerWithKeyCallback", visualizer);
	py::detail::bind_default_constructor<VisualizerWithKeyCallback>(
			visualizer_key);
	visualizer_key
		.def("__repr__", [](const VisualizerWithKeyCallback &vis) {
			return std::string("VisualizerWithKeyCallback with name ") + 
					vis.GetWindowName();
		})
		.def("RegisterKeyCallback",
				&VisualizerWithKeyCallback::RegisterKeyCallback,
				"Function to register a callback function for a key press event",
				"key"_a, "callback_func"_a);
}

void pybind_visualizer_method(py::module &m)
{
}
