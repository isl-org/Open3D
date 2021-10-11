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

#include "open3d/core/TensorFunction.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_tensor_function(py::module& m) {
    m.def(
            "append",
            [](const Tensor& self, const Tensor& values,
               const utility::optional<int64_t> axis) {
                if (axis.has_value()) {
                    return core::Append(self, values, axis);
                }
                return core::Append(self, values);
            },
            R"(Appends the `values` tensor to the `self` tensor, along the 
given axis and returns a new tensor. Both the tensors must have same data-type
device, and number of dimentions. All dimensions must be the same, except the
dimension along the axis the tensors are to be appended. 

This is the same as NumPy's semantics:
- https://numpy.org/doc/stable/reference/generated/numpy.append.html

Returns:
    A copy of the `self` tensor with `values` appended to axis. Note that 
    append does not occur in-place: a new array is allocated and filled. 
    If axis is null, out is a flattened tensor.

Example:
    >>> o3d.core.append([[0, 1], [2, 3]], [[4, 5]], axis = 0)
    [[0 1],
     [2 3],
     [4 5]]
    Tensor[shape={3, 2}, stride={2, 1}, Int64, CPU:0, 0x55555abc6b00]
 
    >>> o3d.core.append([[0, 1], [2, 3]], [[4, 5]])
    [0 1 2 3 4 5]
    Tensor[shape={6}, stride={1}, Int64, CPU:0, 0x55555abc6b70])",
            "self"_a, "values"_a, "axis"_a = py::none());
}

}  // namespace core
}  // namespace open3d
