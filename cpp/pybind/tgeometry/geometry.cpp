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

#include "open3d/tgeometry/Geometry.h"

#include "pybind/docstring.h"
#include "pybind/tgeometry/geometry.h"

namespace open3d {
namespace tgeometry {

void pybind_geometry_class(py::module& m) {
    py::class_<Geometry, PyGeometry<Geometry>, std::unique_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");

    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.")
            .def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
}

void pybind_geometry(py::module& m) {
    py::module m_submodule = m.def_submodule("tgeometry");

    pybind_geometry_class(m_submodule);
    pybind_tensorlistmap(m_submodule);
    pybind_pointcloud(m_submodule);
}

}  // namespace tgeometry
}  // namespace open3d
