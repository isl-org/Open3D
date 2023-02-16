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

#include "open3d/core/SizeVector.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_size_vector(py::module& m) {
    // bind_vector takes care of most common methods for Python list.
    auto sv = py::bind_vector<SizeVector>(
            m, "SizeVector",
            "A vector of integers for specifying shape, strides, etc.");

    // In Python, We want (3), (3,), [3] and [3,] to represent the same thing.
    // The following are all equivalent to core::SizeVector({3}):
    // - o3d.core.SizeVector(3)     # int
    // - o3d.core.SizeVector((3))   # int, not tuple
    // - o3d.core.SizeVector((3,))  # tuple
    // - o3d.core.SizeVector([3])   # list
    // - o3d.core.SizeVector([3,])  # list
    //
    // Difference between C++ and Python:
    // - o3d.core.SizeVector(3) creates a 1-D SizeVector: {3}.
    // - core::SizeVector(3) creates a 3-D SizeVector: {0, 0, 0}.
    //
    // The API difference is due to the NumPy convention which allows integer to
    // represent a 1-element tuple, and the C++ constructor for vectors.
    sv.def(py::init([](int64_t i) { return SizeVector({i}); }));
    py::implicitly_convertible<py::int_, SizeVector>();

    // Allows tuple and list implicit conversions to SizeVector.
    py::implicitly_convertible<py::tuple, SizeVector>();
    py::implicitly_convertible<py::list, SizeVector>();

    auto dsv = py::bind_vector<DynamicSizeVector>(
            m, "DynamicSizeVector",
            "A vector of integers for specifying shape, strides, etc. Some "
            "elements can be None.");
    dsv.def("__repr__",
            [](const DynamicSizeVector& dsv) { return dsv.ToString(); });
}

}  // namespace core
}  // namespace open3d
