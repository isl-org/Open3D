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
    py::enum_<core::Dtype>(m, "Dtype", "Open3D data types.")
            .value("Undefined", core::Dtype::Undefined)
            .value("Float32", core::Dtype::Float32)
            .value("Float64", core::Dtype::Float64)
            .value("Int32", core::Dtype::Int32)
            .value("Int64", core::Dtype::Int64)
            .value("UInt8", core::Dtype::UInt8)
            .value("UInt16", core::Dtype::UInt16)
            .value("Bool", core::Dtype::Bool)
            .export_values();

    py::class_<core::DtypeUtil> dtype_util(m, "DtypeUtil", "Dtype utilities.");
    dtype_util.def(py::init<>()).def("byte_size", &core::DtypeUtil::ByteSize);
}

}  // namespace open3d
