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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace core {

/// \brief Concatenates the list of tensors in their order, along the given
/// axis into a new tensor. All the tensors must have same data-type,
/// device, and number of dimensions. All dimensions must be the same,
/// except the dimension along the axis the tensors are to be concatenated.
/// Using Concatenate for a single tensor, the tensor is split along its
/// first dimension (length), and concatenated along the axis.
///
/// This is the same as NumPy's semantics:
/// - https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
///
/// Example:
/// \code{.cpp}
/// Tensor a = Tensor::Init<int64_t>({{0, 1}, {2, 3}});
/// Tensor b = Tensor::Init<int64_t>({{4, 5}});
/// Tensor c = Tensor::Init<int64_t>({{6, 7}});
/// Tensor output = core::Concatenate({a, b, c}, 0);
/// // output:
/// //  [[0 1],
/// //   [2 3],
/// //   [4 5],
/// //   [6 7]]
/// //  Tensor[shape={4, 2}, stride={2, 1}, Int64, CPU:0, 0x55555abc6b00]
///
/// a = core::Tensor::Init<float>(
///         {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}}, device);
/// output = core::Concatenate({a}, 1);
/// //  output:
/// //  [[0, 1, 4, 5, 8, 9],
/// //   [2, 3, 6, 7, 10, 11]]
/// //  Tensor[shape={2, 6}, stride={6, 1}, Int64, CPU:0, 0x55555abc6b00]
/// \endcode
///
/// \param tensors Vector of tensors to be concatenated. If only one tensor is
/// present, the tensor is split along its first dimension (length), and
/// concatenated along the axis.
/// \param axis [optional] The axis along which values are concatenated.
/// [Default axis is 0].
/// \return A new tensor with the values of list of tensors
/// concatenated in order, along the given axis.
Tensor Concatenate(const std::vector<Tensor>& tensors,
                   const utility::optional<int64_t>& axis = 0);

/// \brief Appends the two tensors, along the given axis into a new tensor.
/// Both the tensors must have same data-type, device, and number of
/// dimensions. All dimensions must be the same, except the dimension along
/// the axis the tensors are to be appended.
///
/// This is the same as NumPy's semantics:
/// - https://numpy.org/doc/stable/reference/generated/numpy.append.html
///
/// Example:
/// \code{.cpp}
/// Tensor a = Tensor::Init<int64_t>({{0, 1}, {2, 3}});
/// Tensor b = Tensor::Init<int64_t>({{4, 5}});
/// Tensor t1 = core::Append(a, b, 0);
/// // t1:
/// //  [[0 1],
/// //   [2 3],
/// //   [4 5]]
/// //  Tensor[shape={3, 2}, stride={2, 1}, Int64, CPU:0, 0x55555abc6b00]
///
/// Tensor t2 = core::Append(a, b);
/// // t2:
/// //  [0 1 2 3 4 5]
/// //  Tensor[shape={6}, stride={1}, Int64, CPU:0, 0x55555abc6b70]
/// \endcode
///
/// \param self Values are appended to a copy of this tensor.
/// \param other Values of this tensor is appended to the `self`.
/// \param axis [optional] The axis along which values are appended. If axis
/// is not given, both tensors are flattened before use.
/// \return A copy of `tensor` with `values` appended to axis. Note that
/// append does not occur in-place: a new array is allocated and filled. If
/// axis is None, out is a flattened tensor.
Tensor Append(const Tensor& self,
              const Tensor& other,
              const utility::optional<int64_t>& axis = utility::nullopt);

/// \brief Computes the element-wise maximum of input and other. The tensors
/// must have same data type and device.
///
/// If input.GetShape() != other.GetShape(), then they will be broadcasted to a
/// common shape (which becomes the shape of the output).
///
/// \param input The input tensor.
/// \param other The second input tensor.
Tensor Maximum(const Tensor& input, const Tensor& other);

/// \brief Computes the element-wise minimum of input and other. The tensors
/// must have same data type and device.
///
/// If input.GetShape() != other.GetShape(), then they will be broadcasted to a
/// common shape (which becomes the shape of the output).
///
/// \param input The input tensor.
/// \param other The second input tensor.
Tensor Minimum(const Tensor& input, const Tensor& other);

}  // namespace core
}  // namespace open3d
