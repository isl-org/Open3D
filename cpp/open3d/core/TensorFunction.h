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

/// \brief Appends the two tensors, along the given axis into a new tensor.
/// Both the tensors must have same data-type, device, and number of
/// dimentions. All dimensions must be the same, except the dimension along
/// the axis the tensors are to be appended.
///
/// This is the same as NumPy's semantics:
/// - https://numpy.org/doc/stable/reference/generated/numpy.append.html
///
/// Example:
/// \code{.cpp}
/// Tensor a = Tensor::Init<int64_t>({0, 1}, {2, 3});
/// Tensor b = Tensor::Init<int64_t>({4, 5});
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
              const utility::optional<int64_t> axis = utility::nullopt);

void RunSYCLDemo();

}  // namespace core
}  // namespace open3d
