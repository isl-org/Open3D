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

#include "py3d_core.h"

#include <Core/Utility/Console.h>
using namespace three;

void pybind_console(py::module &m)
{
	py::enum_<VerbosityLevel>(m, "VerbosityLevel", py::arithmetic())
		.value("Error", VerbosityLevel::VerboseError)
		.value("Warning", VerbosityLevel::VerboseWarning)
		.value("Info", VerbosityLevel::VerboseInfo)
		.value("Debug", VerbosityLevel::VerboseDebug)
		.value("Always", VerbosityLevel::VerboseAlways)
		.export_values();
	m.def("set_verbosity_level", &SetVerbosityLevel,
			"Set global verbosity level of Open3D (py3d)",
			py::arg("verbosity_level"));
	m.def("get_verbosity_level", &GetVerbosityLevel,
			"Get global verbosity level of Open3D (py3d)");
}
