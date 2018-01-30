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

#include "py3d_visualization.h"
#include "py3d_visualization_trampoline.h"

#include <Visualization/Visualizer/RenderOption.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>
using namespace three;

void pybind_renderoption(py::module &m)
{
	py::class_<RenderOption, std::shared_ptr<RenderOption>>
			renderoption(m, "RenderOption");
	py::detail::bind_default_constructor<RenderOption>(renderoption);
	renderoption
		.def("__repr__", [](const RenderOption &vc) {
			return std::string("RenderOption");
		})
		.def("load_from_json", [](RenderOption &ro, const std::string &filename) {
			ReadIJsonConvertible(filename, ro);
		}, "Function to load RenderOption from a JSON file", "filename"_a)
		.def("save_to_json", [](RenderOption &ro, const std::string &filename) {
			WriteIJsonConvertible(filename, ro);
		}, "Function to save RenderOption to a JSON file", "filename"_a)
		.def_readwrite("background_color", &RenderOption::background_color_)
		.def_readwrite("light_on", &RenderOption::light_on_);
}

void pybind_renderoption_method(py::module &m)
{
}
