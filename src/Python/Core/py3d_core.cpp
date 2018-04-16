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

#include "py3d_core.h"

void pybind_core_classes(py::module &m)
{
	pybind_console(m);
	pybind_geometry(m);
	pybind_pointcloud(m);
	pybind_trianglemesh(m);
	pybind_image(m);
	pybind_kdtreeflann(m);
	pybind_feature(m);
	pybind_camera(m);
	pybind_registration(m);
	pybind_odometry(m);
	pybind_globaloptimization(m);
	pybind_integration(m);
}

void pybind_core_methods(py::module &m)
{
	pybind_pointcloud_methods(m);
	pybind_trianglemesh_methods(m);
	pybind_image_methods(m);
	pybind_feature_methods(m);
	pybind_camera_methods(m);
	pybind_registration_methods(m);
	pybind_odometry_methods(m);
	pybind_globaloptimization_methods(m);
	pybind_integration_methods(m);
}
