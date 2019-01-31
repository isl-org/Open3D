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

#include "open3d_core.h"

void pybind_core(py::module &m) {
    py::module m_camera = m.def_submodule("camera");
    py::module m_geometry = m.def_submodule("geometry");
    py::module m_odometry = m.def_submodule("odometry");
    py::module m_registration = m.def_submodule("registration");
    py::module m_integration = m.def_submodule("integration");
    py::module m_utility = m.def_submodule("utility");

    pybind_camera(m_camera);

    pybind_console(m_utility);

    pybind_geometry(m_geometry);
    pybind_pointcloud(m_geometry);
    pybind_voxelgrid(m_geometry);
    pybind_lineset(m_geometry);
    pybind_trianglemesh(m_geometry);
    pybind_image(m_geometry);
    pybind_kdtreeflann(m_geometry);

    pybind_feature(m_registration);
    pybind_registration(m_registration);
    pybind_global_optimization(m_registration);
    pybind_colormap_optimization(m_registration);

    pybind_odometry(m_odometry);

    pybind_integration(m_integration);

    pybind_camera_methods(m_camera);

    pybind_pointcloud_methods(m_geometry);
    pybind_voxelgrid_methods(m_geometry);
    pybind_lineset_methods(m_geometry);
    pybind_trianglemesh_methods(m_geometry);
    pybind_image_methods(m_geometry);

    pybind_feature_methods(m_registration);
    pybind_registration_methods(m_registration);
    pybind_global_optimization_methods(m_registration);
    pybind_colormap_optimization_methods(m_registration);

    pybind_odometry_methods(m_odometry);

    pybind_integration_methods(m_integration);
}
