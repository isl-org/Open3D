// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/open3d_pybind.h"

#include "open3d/core/MemoryManagerStatistic.h"
#include "open3d/utility/Logging.h"
#include "pybind/camera/camera.h"
#include "pybind/core/core.h"
#include "pybind/data/dataset.h"
#include "pybind/geometry/geometry.h"
#include "pybind/io/io.h"
#include "pybind/ml/ml.h"
#include "pybind/pipelines/pipelines.h"
#include "pybind/t/t.h"
#include "pybind/utility/utility.h"
#include "pybind/visualization/visualization.h"

namespace open3d {

PYBIND11_MODULE(pybind, m) {
    utility::Logger::GetInstance().SetPrintFunction([](const std::string& msg) {
        py::gil_scoped_acquire acquire;
        py::print(msg);
    });

    m.doc() = "Python binding of Open3D";

    // Check Open3D CXX11_ABI with
    // import open3d as o3d; print(o3d.open3d_pybind._GLIBCXX_USE_CXX11_ABI)
    m.add_object("_GLIBCXX_USE_CXX11_ABI",
                 _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False);

    // The binding order matters: if a class haven't been binded, binding the
    // user of this class will result in "could not convert default argument
    // into a Python object" error.
    utility::pybind_utility_declarations(m);
    camera::pybind_camera_declarations(m);
    core::pybind_core_declarations(m);
    data::pybind_data(m);
    geometry::pybind_geometry_declarations(m);
    t::pybind_t_declarations(m);
    ml::pybind_ml_declarations(m);
    io::pybind_io_declarations(m);
    pipelines::pybind_pipelines_declarations(m);
    visualization::pybind_visualization_declarations(m);

    utility::pybind_utility_definitions(m);
    camera::pybind_camera_definitions(m);
    core::pybind_core_definitions(m);
    geometry::pybind_geometry_definitions(m);
    t::pybind_t_definitions(m);
    ml::pybind_ml_definitions(m);
    io::pybind_io_definitions(m);
    pipelines::pybind_pipelines_definitions(m);
    visualization::pybind_visualization_definitions(m);

    // pybind11 will internally manage the lifetime of default arguments for
    // function bindings. Since these objects will live longer than the memory
    // manager statistics, the latter will report leaks. Reset the statistics to
    // ignore them and transfer the responsibility to pybind11.
    core::MemoryManagerStatistic::GetInstance().Reset();
}

}  // namespace open3d
