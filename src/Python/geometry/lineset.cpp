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

#include "Python/geometry/geometry_trampoline.h"
#include "Python/geometry/geometry.h"

#include <Open3D/Geometry/LineSet.h>

using namespace open3d;

void pybind_lineset(py::module &m) {
    py::class_<geometry::LineSet, PyGeometry3D<geometry::LineSet>,
               std::shared_ptr<geometry::LineSet>, geometry::Geometry3D>
            lineset(m, "LineSet", "LineSet");
    py::detail::bind_default_constructor<geometry::LineSet>(lineset);
    py::detail::bind_copy_functions<geometry::LineSet>(lineset);
    lineset.def("__repr__",
                [](const geometry::LineSet &lineset) {
                    return std::string("geometry::LineSet with ") +
                           std::to_string(lineset.lines_.h_data.size()) +
                           " lines.";
                })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_points", &geometry::LineSet::HasPoints)
            .def("has_lines", &geometry::LineSet::HasLines)
            .def("has_colors", &geometry::LineSet::HasColors)
            .def("normalize_normals", &geometry::LineSet::GetLineCoordinate)
            .def_readwrite("points", &geometry::LineSet::points_.h_data)
            .def_readwrite("lines", &geometry::LineSet::lines_.h_data)
            .def_readwrite("colors", &geometry::LineSet::colors_.h_data);
}
