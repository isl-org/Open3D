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

#include "open3d_pybind/core/container.h"
#include "open3d_pybind/docstring.h"
#include "open3d_pybind/open3d_pybind.h"

#include "Open3D/Core/Dtype.h"

using namespace open3d;

void pybind_core_dtype(py::module &m) {
    py::enum_<Dtype>(m, "Dtype")
            .value("Undefined", Dtype::Undefined)
            .value("Float32", Dtype::Float32)
            .value("Float64", Dtype::Float64)
            .value("Int32", Dtype::Int32)
            .value("Int64", Dtype::Int64)
            .value("UInt8", Dtype::UInt8)
            .value("Bool", Dtype::Bool)
            .export_values();

    py::class_<DtypeUtil> dtype_util(m, "DtypeUtil");
    dtype_util.def(py::init<>()).def("byte_size", &DtypeUtil::ByteSize);
}
