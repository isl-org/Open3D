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

#include "pybind/open3d_pybind.h"

#include "open3d/utility/Console.h"
#include "pybind/camera/camera.h"
#include "pybind/core/core.h"
#include "pybind/geometry/geometry.h"
#include "pybind/io/io.h"
#include "pybind/ml/ml.h"
#include "pybind/pipelines/pipelines.h"
#include "pybind/t/t.h"
#include "pybind/utility/utility.h"
#include "pybind/visualization/visualization.h"

namespace open3d {

PYBIND11_MODULE(pybind, m) {
    open3d::utility::Logger::i().print_fcn_ = [](const std::string& msg) {
        py::gil_scoped_acquire acquire;
        py::print(msg);
    };

    m.doc() = "Python binding of Open3D";

    // Check Open3D CXX11_ABI with
    // import open3d as o3d; print(o3d.open3d_pybind._GLIBCXX_USE_CXX11_ABI)
    m.add_object("_GLIBCXX_USE_CXX11_ABI",
                 _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False);

    // The binding order matters: if a class haven't been binded, binding the
    // user of this class will result in "could not convert default argument
    // into a Python object" error.
    utility::pybind_utility(m);

    camera::pybind_camera(m);
    core::pybind_core(m);
    geometry::pybind_geometry(m);
    t::pybind_t(m);
    ml::pybind_ml(m);
    io::pybind_io(m);
    pipelines::pybind_pipelines(m);
    visualization::pybind_visualization(m);
}

}  // namespace open3d
