// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Logging.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {

void pybind_logging_declarations(py::module& m) {
    py::native_enum<VerbosityLevel>(m, "VerbosityLevel", "enum.IntEnum",
                                    "Enum class for VerbosityLevel.")
            .value("Error", VerbosityLevel::Error)
            .value("Warning", VerbosityLevel::Warning)
            .value("Info", VerbosityLevel::Info)
            .value("Debug", VerbosityLevel::Debug)
            .export_values()
            .finalize();
    py::class_<VerbosityContextManager> verbosity_context_manager(
            m, "VerbosityContextManager",
            "A context manager to "
            "temporally change the "
            "verbosity level of Open3D");
}
void pybind_logging_definitions(py::module& m) {
    m.def("set_verbosity_level", &SetVerbosityLevel,
          "Set global verbosity level of Open3D", py::arg("verbosity_level"));
    docstring::FunctionDocInject(
            m, "set_verbosity_level",
            {{"verbosity_level",
              "Messages with equal or less than ``verbosity_level`` verbosity "
              "will be printed."}});

    m.def("get_verbosity_level", &GetVerbosityLevel,
          "Get global verbosity level of Open3D");
    docstring::FunctionDocInject(m, "get_verbosity_level");

    m.def("reset_print_function", []() {
        utility::LogInfo("Resetting default logger to print to terminal.");
        utility::Logger::GetInstance().ResetPrintFunction();
    });
    auto verbosity_context_manager =
            static_cast<py::class_<VerbosityContextManager>>(
                    m.attr("VerbosityContextManager"));
    verbosity_context_manager
            .def(py::init<VerbosityLevel>(),
                 "Create a VerbosityContextManager with a given VerbosityLevel",
                 "level"_a)
            .def(
                    "__enter__",
                    [&](VerbosityContextManager& cm) { cm.Enter(); },
                    "Enter the context manager")
            .def(
                    "__exit__",
                    [&](VerbosityContextManager& cm, pybind11::object exc_type,
                        pybind11::object exc_value,
                        pybind11::object traceback) { cm.Exit(); },
                    "Exit the context manager");
}

}  // namespace utility
}  // namespace open3d
