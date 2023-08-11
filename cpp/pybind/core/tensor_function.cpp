// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/TensorFunction.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

void pybind_core_tensor_function(py::module& m) {
    m.def(
            "concatenate",
            [](const std::vector<Tensor>& tensors,
               const utility::optional<int64_t>& axis) {
                if (axis.has_value()) {
                    return core::Concatenate(tensors, axis);
                }
                return core::Concatenate(tensors);
            },
            R"(Concatenates the list of tensors in their order, along the given
axis into a new tensor. All the tensors must have same data-type, device, and
number of dimensions. All dimensions must be the same, except the dimension
along the axis the tensors are to be concatenated.
Using Concatenate for a single tensor, the tensor is split along its first 
dimension (length), and concatenated along the axis.

This is the same as NumPy's semantics:
- https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

Returns:
     A new tensor with the values of list of tensors concatenated in order,
     along the given axis.

Example:
    >>> a = o3d.core.Tensor([[0, 1], [2, 3]])
    >>> b = o3d.core.Tensor([[4, 5]])
    >>> c = o3d.core.Tensor([[6, 7])
    >>> o3d.core.concatenate((a, b, c), 0)
    [[0 1],
     [2 3],
     [4 5],
     [6 7],
     [8 9]]
    Tensor[shape={5, 2}, stride={2, 1}, Int64, CPU:0, 0x55b454b09390])",
            "tensors"_a, "axis"_a = 0);

    m.def(
            "append",
            [](const Tensor& self, const Tensor& values,
               const utility::optional<int64_t>& axis) {
                if (axis.has_value()) {
                    return core::Append(self, values, axis);
                }
                return core::Append(self, values);
            },
            R"(Appends the `values` tensor to the `self` tensor, along the 
given axis and returns a new tensor. Both the tensors must have same data-type
device, and number of dimensions. All dimensions must be the same, except the
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

    m.def("maximum", &core::Maximum,
          R"(Computes the element-wise maximum of input and other. The tensors 
must have same data type and device.
If input.GetShape() != other.GetShape(), then they will be broadcasted to a
common shape (which becomes the shape of the output).)",
          "input"_a, "other"_a);
    m.def("minimum", &core::Minimum,
          R"(Computes the element-wise minimum of input and other. The tensors 
must have same data type and device.
If input.GetShape() != other.GetShape(), then they will be broadcasted to a
common shape (which becomes the shape of the output).)",
          "input"_a, "other"_a);
}

}  // namespace core
}  // namespace open3d
