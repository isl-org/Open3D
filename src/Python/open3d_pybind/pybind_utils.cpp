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

#include "open3d_pybind/pybind_utils.h"

#include <string>

#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Tensor.h"

#include "open3d_pybind/open3d_pybind.h"

namespace open3d {
namespace pybind_utils {

Dtype ArrayFormatToDtype(const std::string& format) {
    if (format == py::format_descriptor<float>::format()) {
        return Dtype::Float32;
    } else if (format == py::format_descriptor<double>::format()) {
        return Dtype::Float64;
    } else if (format == py::format_descriptor<int32_t>::format()) {
        return Dtype::Int32;
    } else if (format == py::format_descriptor<int64_t>::format()) {
        return Dtype::Int64;
    } else if (format == py::format_descriptor<uint8_t>::format()) {
        return Dtype::UInt8;
    } else if (format == py::format_descriptor<bool>::format()) {
        return Dtype::Bool;
    } else {
        utility::LogError("Unsupported data type.");
    }
}

std::string DtypeToArrayFormat(const Dtype& dtype) {
    if (dtype == Dtype::Float32) {
        return py::format_descriptor<float>::format();
    } else if (dtype == Dtype::Float64) {
        return py::format_descriptor<double>::format();
    } else if (dtype == Dtype::Int32) {
        return py::format_descriptor<int32_t>::format();
    } else if (dtype == Dtype::Int64) {
        return py::format_descriptor<int64_t>::format();
    } else if (dtype == Dtype::UInt8) {
        return py::format_descriptor<uint8_t>::format();
    } else if (dtype == Dtype::Bool) {
        return py::format_descriptor<bool>::format();
    } else {
        utility::LogError("Unsupported data type.");
    }
}

}  // namespace pybind_utils
}  // namespace open3d
