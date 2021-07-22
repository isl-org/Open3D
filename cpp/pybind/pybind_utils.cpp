// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

core::Dtype ArrayFormatToDtype(const std::string& format, size_t byte_size) {
    // In general, format characters follows the standard:
    // https://docs.python.org/3/library/struct.html#format-characters
    //
    // However, some integer dtypes have aliases. E.g. "l" can be 4 bytes or 8
    // bytes depending on the OS. To be safe, we always check the byte size.
    if (format == py::format_descriptor<float>::format() && byte_size == 4)
        return core::Float32;
    if (format == py::format_descriptor<double>::format() && byte_size == 8)
        return core::Float64;
    if (format == py::format_descriptor<int8_t>::format() && byte_size == 1)
        return core::Int8;
    if (format == py::format_descriptor<int16_t>::format() && byte_size == 2)
        return core::Int16;
    if ((format == py::format_descriptor<int32_t>::format() || format == "i" ||
         format == "l") &&
        byte_size == 4)
        return core::Int32;
    if ((format == py::format_descriptor<int64_t>::format() || format == "q" ||
         format == "l") &&
        byte_size == 8)
        return core::Int64;
    if (format == py::format_descriptor<uint8_t>::format() && byte_size == 1)
        return core::UInt8;
    if (format == py::format_descriptor<uint16_t>::format() && byte_size == 2)
        return core::UInt16;
    if ((format == py::format_descriptor<uint32_t>::format() ||
         format == "L") &&
        byte_size == 4)
        return core::UInt32;
    if ((format == py::format_descriptor<uint64_t>::format() ||
         format == "L") &&
        byte_size == 8)
        return core::UInt64;
    if (format == py::format_descriptor<bool>::format() && byte_size == 1)
        return core::Bool;
    utility::LogError(
            "ArrayFormatToDtype: unsupported python array format {} with "
            "byte_size {}.",
            format, byte_size);
    return core::Undefined;
}

std::string DtypeToArrayFormat(const core::Dtype& dtype) {
    if (dtype == core::Float32) return py::format_descriptor<float>::format();
    if (dtype == core::Float64) return py::format_descriptor<double>::format();
    if (dtype == core::Int8) return py::format_descriptor<int8_t>::format();
    if (dtype == core::Int16) return py::format_descriptor<int16_t>::format();
    if (dtype == core::Int32) return py::format_descriptor<int32_t>::format();
    if (dtype == core::Int64) return py::format_descriptor<int64_t>::format();
    if (dtype == core::UInt8) return py::format_descriptor<uint8_t>::format();
    if (dtype == core::UInt16) return py::format_descriptor<uint16_t>::format();
    if (dtype == core::UInt32) return py::format_descriptor<uint32_t>::format();
    if (dtype == core::UInt64) return py::format_descriptor<uint64_t>::format();
    if (dtype == core::Bool) return py::format_descriptor<bool>::format();
    utility::LogError("Unsupported data type.");
    return std::string();
}

}  // namespace pybind_utils
}  // namespace open3d
