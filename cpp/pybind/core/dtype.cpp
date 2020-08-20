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

#include "open3d/core/Dtype.h"

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {

void pybind_core_dtype(py::module &m) {
    py::class_<core::Dtype, std::shared_ptr<core::Dtype>> dtype(
            m, "Dtype", "Open3D data types.");
    dtype.def_readonly_static("Undefined", &core::Undefined)
            .def_readonly_static("Float32", &core::Float32)
            .def_readonly_static("Float64", &core::Float64)
            .def_readonly_static("Int32", &core::Int32)
            .def_readonly_static("Int64", &core::Int64)
            .def_readonly_static("UInt8", &core::UInt8)
            .def_readonly_static("UInt16", &core::UInt16)
            .def_readonly_static("Bool", &core::Bool)
            .def("byte_size", &core::Dtype::ByteSize)
            .def("__eq__", &core::Dtype::operator==)
            .def("__ene__", &core::Dtype::operator!=)
            .def("__repr__", &core::Dtype::ToString)
            .def("__str__", &core::Dtype::ToString);
}

}  // namespace open3d
