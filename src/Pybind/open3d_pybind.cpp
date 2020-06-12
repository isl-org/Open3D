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

#include "Pybind/open3d_pybind.h"
#include "Pybind/camera/camera.h"
#include "Pybind/color_map/color_map.h"
#include "Pybind/core/container.h"
#include "Pybind/geometry/geometry.h"
#include "Pybind/integration/integration.h"
#include "Pybind/io/io.h"
#include "Pybind/odometry/odometry.h"
#include "Pybind/registration/registration.h"
#include "Pybind/utility/utility.h"
#include "Pybind/visualization/visualization.h"

#include "Open3D/Utility/Console.h"

namespace open3d {

PYBIND11_MODULE(pybind, m) {
    open3d::utility::Logger::i().print_fcn_ = [](const std::string& msg) {
        py::print(msg);
    };

    m.doc() = "Python binding of Open3D";

    // Check open3d CXX11_ABI with
    // import open3d; print(open3d.open3d._GLIBCXX_USE_CXX11_ABI)
    m.add_object("_GLIBCXX_USE_CXX11_ABI",
                 _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False);

    // Register this first, other submodule (e.g. odometry) might depend on this
    pybind_utility(m);
    pybind_core(m);
    pybind_camera(m);
    pybind_color_map(m);
    pybind_geometry(m);
    pybind_integration(m);
    pybind_io(m);
    pybind_registration(m);
    pybind_odometry(m);
    pybind_visualization(m);
}

}  // namespace open3d
