// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/t/geometry/TensorMap.h"

#include "open3d/t/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_tensormap(py::module& m) {
    // Bind to the generic dictionary interface such that it works the same as a
    // regular dictionay in Python, except that types are enforced. Supported
    // functions include `__bool__`, `__iter__`, `items`, `__getitem__`,
    // `__contains__`, `__delitem__`, `__len__` and map assignment.
    auto tm = py::bind_map<TensorMap>(
            m, "TensorMap", "Map of String to Tensor with a primary key.");

    // Constructors.
    tm.def(py::init<const std::string&>(), "primary_key"_a);
    tm.def(py::init<const std::string&,
                    const std::unordered_map<std::string, core::Tensor>&>(),
           "primary_key"_a, "map_keys_to_tensors"_a);

    // Member functions. Some C++ functions are ignored since the
    // functionalities are already covered in the generic dictionary interface.
    tm.def("get_primary_key", &TensorMap::GetPrimaryKey);
    tm.def("is_size_synchronized", &TensorMap::IsSizeSynchronized);
    tm.def("assert_size_synchronized", &TensorMap::AssertSizeSynchronized);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
