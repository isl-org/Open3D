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

#include "Open3D/Utility/Console.h"
#include "Python/docstring.h"
#include "Python/open3d_pybind.h"

using namespace open3d;

void pybind_console(py::module &m) {
    py::enum_<utility::VerbosityLevel> vl(m, "VerbosityLevel", py::arithmetic(),
                                          "VerbosityLevel");
    vl.value("Off", utility::VerbosityLevel::Off)
            .value("Fatal", utility::VerbosityLevel::Fatal)
            .value("Error", utility::VerbosityLevel::Error)
            .value("Warning", utility::VerbosityLevel::Warning)
            .value("Info", utility::VerbosityLevel::Info)
            .value("Debug", utility::VerbosityLevel::Debug)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    vl.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for VerbosityLevel.";
            }),
            py::none(), py::none(), "");

    m.def("set_verbosity_level", &utility::SetVerbosityLevel,
          "Set global verbosity level of Open3D", py::arg("verbosity_level"));
    docstring::FunctionDocInject(
            m, "set_verbosity_level",
            {{"verbosity_level",
              "Messages with equal or less than ``verbosity_level`` verbosity "
              "will be printed."}});

    m.def("get_verbosity_level", &utility::GetVerbosityLevel,
          "Get global verbosity level of Open3D");
    docstring::FunctionDocInject(m, "get_verbosity_level");
}
