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

#pragma once

#include "Python/open3d_pybind.h"

void pybind_geometry(py::module &m);

void pybind_pointcloud(py::module &m);
void pybind_voxelgrid(py::module &m);
void pybind_lineset(py::module &m);
void pybind_trianglemesh(py::module &m);
void pybind_halfedgetrianglemesh(py::module &m);
void pybind_image(py::module &m);
void pybind_tetramesh(py::module &m);
void pybind_kdtreeflann(py::module &m);
void pybind_pointcloud_methods(py::module &m);
void pybind_voxelgrid_methods(py::module &m);
void pybind_trianglemesh_methods(py::module &m);
void pybind_lineset_methods(py::module &m);
void pybind_image_methods(py::module &m);
void pybind_octree_methods(py::module &m);
void pybind_octree(py::module &m);
void pybind_boundingvolume(py::module &m);
