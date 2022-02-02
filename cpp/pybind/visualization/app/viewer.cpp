// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "pybind/visualization/app/viewer.h"

#include "open3d/visualization/app/Viewer.h"
#include "pybind/docstring.h"

namespace open3d {
namespace visualization {
namespace app {

static void pybind_app_functions(py::module &m) {
    m.def(
            "run_viewer",
            [](const std::vector<std::string> &args) {
                const char **argv = new const char *[args.size()];
                for (size_t it = 0; it < args.size(); it++) {
                    argv[it] = args[it].c_str();
                }
                RunViewer(args.size(), argv);
                delete[] argv;
            },
            "args"_a);

    docstring::FunctionDocInject(
            m, "run_viewer",
            {{"args",
              "List of arguments containing the path of the calling program "
              "(which should be in the same directory as the gui resoruces "
              "folder) and the optional path of the geometry to visualize."}});
}

void pybind_app(py::module &m) {
    py::module m_submodule = m.def_submodule(
            "app", "Functionality for running the open3d viewer.");
    pybind_app_functions(m_submodule);
}

}  // namespace app
}  // namespace visualization
}  // namespace open3d
