// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Logging.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {

void pybind_logging(py::module& m) {
    py::enum_<VerbosityLevel> vl(m, "VerbosityLevel", py::arithmetic(),
                                 "VerbosityLevel");
    vl.value("Error", VerbosityLevel::Error)
            .value("Warning", VerbosityLevel::Warning)
            .value("Info", VerbosityLevel::Info)
            .value("Debug", VerbosityLevel::Debug)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    vl.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for VerbosityLevel.";
            }),
            py::none(), py::none(), "");

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

    py::class_<VerbosityContextManager>(m, "VerbosityContextManager",
                                        "A context manager to "
                                        "temporally change the "
                                        "verbosity level of Open3D")
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
