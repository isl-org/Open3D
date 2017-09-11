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

#include <Visualization/Visualizer/ViewControl.h>
using namespace three;

void pybind_viewcontrol(py::module &m)
{
	py::class_<ViewControl, PyViewControl<>, std::shared_ptr<ViewControl>>
			viewcontrol(m, "ViewControl");
	py::detail::bind_default_constructor<ViewControl>(viewcontrol);
	viewcontrol
		.def("__repr__", [](const ViewControl &vc) {
			return std::string("ViewControl");
		})
		.def("ConvertToPinholeCameraParameters", [](ViewControl &vc) {
			PinholeCameraIntrinsic intrinsic;
			Eigen::Matrix4d extrinsic;
			vc.ConvertToPinholeCameraParameters(intrinsic, extrinsic);
			return std::make_tuple(intrinsic, extrinsic);
		}, "Function to convert ViewControl to PinholeCameraParameters")
		.def("ConvertFromPinholeCameraParameters",
				&ViewControl::ConvertFromPinholeCameraParameters,
				"intrinsic"_a, "extrinsic"_a)
		.def("Scale", &ViewControl::Scale, "Function to process scaling",
				"scale"_a)
		.def("Rotate", &ViewControl::Rotate, "Function to process rotation",
				"x"_a, "y"_a, "xo"_a = 0.0, "yo"_a = 0.0)
		.def("Translate", &ViewControl::Translate,
				"Function to process translation",
				"x"_a, "y"_a, "xo"_a = 0.0, "yo"_a = 0.0);
}

void pybind_viewcontrol_method(py::module &m)
{
}
