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

#include <pybind11/pybind11.h>

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_dtype(py::module &m) {
    py::enum_<Dtype::DtypeCode>(m, "DtypeCode", "Open3D data type codes.")
            .value("Undefined", Dtype::DtypeCode::Undefined)
            .value("Bool", Dtype::DtypeCode::Bool)
            .value("Int", Dtype::DtypeCode::Int)
            .value("UInt", Dtype::DtypeCode::UInt)
            .value("Float", Dtype::DtypeCode::Float)
            .value("Object", Dtype::DtypeCode::Object)
            .export_values();

    py::class_<Dtype, std::shared_ptr<Dtype>> dtype(m, "Dtype",
                                                    "Open3D data types.");
    dtype.def(py::init<Dtype::DtypeCode, int64_t, const std::string &>());
    dtype.def_readonly_static("Undefined", &Dtype::Undefined);
    dtype.def_readonly_static("Float32", &Dtype::Float32);
    dtype.def_readonly_static("Float64", &Dtype::Float64);
    dtype.def_readonly_static("Int8", &Dtype::Int8);
    dtype.def_readonly_static("Int16", &Dtype::Int16);
    dtype.def_readonly_static("Int32", &Dtype::Int32);
    dtype.def_readonly_static("Int64", &Dtype::Int64);
    dtype.def_readonly_static("UInt8", &Dtype::UInt8);
    dtype.def_readonly_static("UInt16", &Dtype::UInt16);
    dtype.def_readonly_static("UInt32", &Dtype::UInt32);
    dtype.def_readonly_static("UInt64", &Dtype::UInt64);
    dtype.def_readonly_static("Bool", &Dtype::Bool);
    dtype.def("byte_size", &Dtype::ByteSize);
    dtype.def("byte_code", &Dtype::GetDtypeCode);
    dtype.def("__eq__", &Dtype::operator==);
    dtype.def("__ene__", &Dtype::operator!=);
    dtype.def("__repr__", &Dtype::ToString);
    dtype.def("__str__", &Dtype::ToString);
}

}  // namespace core
}  // namespace open3d
