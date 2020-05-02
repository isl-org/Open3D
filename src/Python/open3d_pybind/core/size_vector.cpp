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

#include "Open3D/Core/SizeVector.h"

using namespace open3d;

void pybind_core_size_vector(py::module &m) {
    py::class_<SizeVector> size_vector(m, "SizeVector",
                                       "SizeVector is a vector of int64_t for "
                                       "specifying shape, strides and etc.");

    size_vector.def(py::init(
            [](py::array_t<int64_t, py::array::c_style | py::array::forcecast>
                       np_array) {
                py::buffer_info info = np_array.request();
                if (info.ndim != 1) {
                    utility::LogError("SizeVector must be 1-D array.");
                }
                // The buffer is copied to avoid corruption.
                int64_t *start = static_cast<int64_t *>(info.ptr);
                return new SizeVector(start, start + info.shape[0]);
            }));
    size_vector.def("to_string", &SizeVector::ToString);
    size_vector.def("__repr__", [](const SizeVector &size_vector) {
        return size_vector.ToString();
    });
    size_vector.def("__eq__",
                    [](const SizeVector &src, const SizeVector &dst) -> bool {
                        return src == dst;
                    });
}
