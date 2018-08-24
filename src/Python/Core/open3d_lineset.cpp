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
#include "open3d_core_trampoline.h"

#include <Core/Geometry/LineSet.h>
using namespace open3d;

void pybind_lineset(py::module &m)
{
    py::class_<LineSet, PyGeometry3D<LineSet>,
            std::shared_ptr<LineSet>, Geometry3D> lineset(m,
            "LineSet");
    py::detail::bind_default_constructor<LineSet>(lineset);
    py::detail::bind_copy_functions<LineSet>(lineset);
    lineset
        .def("__repr__", [](const LineSet &lineset) {
            return std::string("LineSet with ") +
                    std::to_string(lineset.lines_.size()) + " lines.";
        })
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def("has_points", &LineSet::HasPoints)
        .def("has_lines", &LineSet::HasLines)
        .def("has_colors", &LineSet::HasColors)
        .def("normalize_normals", &LineSet::GetLineCoordinate)
        .def_readwrite("points", &LineSet::points_)
        .def_readwrite("lines", &LineSet::lines_)
        .def_readwrite("colors", &LineSet::colors_);
}

void pybind_lineset_methods(py::module &m)
{
    
}
