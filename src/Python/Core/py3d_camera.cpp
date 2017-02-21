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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Camera/PinholeCameraIntrinsic.h>
#include <Core/Camera/PinholeCameraTrajectory.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>
using namespace three;

void pybind_camera(py::module &m)
{
	py::class_<PinholeCameraIntrinsic> pinhole_intr(m,
			"PinholeCameraIntrinsic", py::metaclass());
	py::detail::bind_default_constructor<PinholeCameraIntrinsic>(pinhole_intr);
	py::detail::bind_copy_functions<PinholeCameraIntrinsic>(pinhole_intr);
	pinhole_intr.def("__init__", [](PinholeCameraIntrinsic &c, int w, int h,
			double fx, double fy, double cx, double cy) {
		new (&c)PinholeCameraIntrinsic(w, h, fx, fy, cx, cy);
	}, "width"_a, "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a);
	pinhole_intr
		.def("SetIntrinsics", &PinholeCameraIntrinsic::SetIntrinsics,
				"width"_a, "height"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a)
		.def("GetFocalLength", &PinholeCameraIntrinsic::GetFocalLength)
		.def("GetPrincipalPoint", &PinholeCameraIntrinsic::GetPrincipalPoint)
		.def("GetSkew", &PinholeCameraIntrinsic::GetSkew)
		.def("IsValid", &PinholeCameraIntrinsic::IsValid)
		.def_readwrite("width", &PinholeCameraIntrinsic::width_)
		.def_readwrite("height", &PinholeCameraIntrinsic::height_)
		.def_readwrite("intrinsic_matrix",
				&PinholeCameraIntrinsic::intrinsic_matrix_)
		.def_readonly_static("PrimeSenseDefault",
				&PinholeCameraIntrinsic::PrimeSenseDefault)
		.def("__repr__", [](const PinholeCameraIntrinsic &c) {
			return std::string("PinholeCameraIntrinsic with width = ") +
					std::to_string(c.width_) + std::string(" and height = ") +
					std::to_string(c.height_) + 
					std::string(".\nAccess intrinsics with intrinsic_matrix.");
		});
}

void pybind_camera_methods(py::module &m)
{
	m.def("ReadPinholeCameraIntrinsic", [](const std::string &filename) {
		PinholeCameraIntrinsic intrinsic;
		ReadIJsonConvertible(filename, intrinsic);
		return intrinsic;
	}, "Function to read PinholeCameraIntrinsic from file", "filename"_a);
	m.def("WritePinholeCameraIntrinsic", [](const std::string &filename,
			const PinholeCameraIntrinsic &intrinsic) {
		return WriteIJsonConvertible(filename, intrinsic);
	}, "Function to write PinholeCameraIntrinsic to file", "filename"_a,
			"intrinsic"_a);
}