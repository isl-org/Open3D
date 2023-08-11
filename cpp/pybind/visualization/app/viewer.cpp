// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
              "(which should be in the same directory as the gui resources "
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
