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

#include "pybind/pybind_utils.h"

#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace pybind_utils {

core::Dtype ArrayFormatToDtype(const std::string& format) {
    if (format == py::format_descriptor<float>::format()) {
        return core::Dtype::Float32;
    } else if (format == py::format_descriptor<double>::format()) {
        return core::Dtype::Float64;
    } else if (format == py::format_descriptor<int32_t>::format()) {
        return core::Dtype::Int32;
    } else if (format == py::format_descriptor<int64_t>::format()) {
        return core::Dtype::Int64;
    } else if (format == py::format_descriptor<uint8_t>::format()) {
        return core::Dtype::UInt8;
    } else if (format == py::format_descriptor<bool>::format()) {
        return core::Dtype::Bool;
    } else {
        utility::LogError("Unsupported data type.");
    }
}

std::string DtypeToArrayFormat(const core::Dtype& dtype) {
    if (dtype == core::Dtype::Float32) {
        return py::format_descriptor<float>::format();
    } else if (dtype == core::Dtype::Float64) {
        return py::format_descriptor<double>::format();
    } else if (dtype == core::Dtype::Int32) {
        return py::format_descriptor<int32_t>::format();
    } else if (dtype == core::Dtype::Int64) {
        return py::format_descriptor<int64_t>::format();
    } else if (dtype == core::Dtype::UInt8) {
        return py::format_descriptor<uint8_t>::format();
    } else if (dtype == core::Dtype::Bool) {
        return py::format_descriptor<bool>::format();
    } else {
        utility::LogError("Unsupported data type.");
    }
}

}  // namespace pybind_utils
}  // namespace open3d
