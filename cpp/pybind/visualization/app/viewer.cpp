// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/visualization/app/viewer.h"

#include "open3d/visualization/app/Viewer.h"
#include "pybind/docstring.h"

namespace open3d {
namespace visualization {
namespace app {

void pybind_app_declarations(py::module &m) {
    py::module m_app = m.def_submodule(
            "app", "Functionality for running the open3d viewer.");
}
void pybind_app_definitions(py::module &m) {
    auto m_app = static_cast<py::module>(m.attr("app"));
    m_app.def(
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
            m_app, "run_viewer",
            {{"args",
              "List of arguments containing the path of the calling program "
              "(which should be in the same directory as the gui resources "
              "folder) and the optional path of the geometry to visualize."}});
}

}  // namespace app
}  // namespace visualization
}  // namespace open3d
