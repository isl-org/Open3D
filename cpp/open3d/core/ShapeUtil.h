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

#pragma once

#include "open3d/core/Dispatch.h"
#include "open3d/core/SizeVector.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {

class Tensor;

namespace shape_util {

/// \brief Returns true if two shapes are compatible for broadcasting.
///
/// E.g. IsCompatibleBroadcastShape({3, 1, 2}, {5, 1}) -> true
///      IsCompatibleBroadcastShape({3, 1, 2}, {5, 3}) -> false
/// \param l_shape Shape of the left-hand-side Tensor.
/// \param r_shape Shape of the left-hand-side Tensor.
/// \return Returns true if \p l_shape and \p r_shape are compatible for
/// broadcasting.
bool IsCompatibleBroadcastShape(const SizeVector& l_shape,
                                const SizeVector& r_shape);

/// \brief Returns the broadcasted shape of two shapes.
///
/// E.g. BroadcastedShape({3, 1, 2}, {5, 1}) -> {3, 5, 2}
///      BroadcastedShape({3, 1, 2}, {5, 3}) -> Exception
/// \param l_shape Shape of the left-hand-side Tensor.
/// \param r_shape Shape of the left-hand-side Tensor.
/// \return The broadcasted shape.
SizeVector BroadcastedShape(const SizeVector& l_shape,
                            const SizeVector& r_shape);

/// \brief Returns true if \p src_shape can be brocasted to \p dst_shape.
///
/// E.g. CanBeBrocastedToShape({1, 2}, {3, 5, 2}) -> true
///      CanBeBrocastedToShape({1, 2}, {3, 5, 3}) -> false
/// \param src_shape Source tensor shape.
/// \param dst_shape Destination tensor shape.
/// \return Returns true if \p src_shape can be brocasted to \p dst_shape.
bool CanBeBrocastedToShape(const SizeVector& src_shape,
                           const SizeVector& dst_shape);

/// \brief Returns the shape after reduction.
///
/// E.g. CanBeBrocastedToShape({1, 2}, {3, 5, 2}) -> true
///      CanBeBrocastedToShape({1, 2}, {3, 5, 3}) -> false
/// \param src_shape shape to reduce
/// \param dims A list of dimensions to be reduced.
/// \param keepdim If true, the reduced dims will be retained as size 1.
SizeVector ReductionShape(const SizeVector& src_shape,
                          const SizeVector& dims,
                          bool keepdim);

/// \brief Wrap around negative \p dim.
///
/// E.g. If max_dim == 5, dim -1 will be converted to 4.
///
/// \param dim Dimension index
/// \param max_dim Maximum dimension index
/// \param inclusive Set to true to allow dim == max_dim. E.g. for slice
///        T[start:end], we allow end == max_dim.
int64_t WrapDim(int64_t dim, int64_t max_dim, bool inclusive = false);

/// Infers the size of a dim with size -1, if it exists. Also checks that new
/// shape is compatible with the number of elements.
///
/// E.g. Shape({2, -1, 4}) with num_elemnts 24, will be inferred as {2, 3, 4}.
///
/// Ref: PyTorch's aten/src/ATen/InferSize.h
SizeVector InferShape(SizeVector shape, int64_t num_elements);

/// Concatenate two shapes.
SizeVector Concat(const SizeVector& l_shape, const SizeVector& r_shape);

/// Returns a SizeVector of {0, 1, ..., n - 1}, similar to std::iota.
SizeVector Iota(int64_t n);

/// Compute default strides for a shape when a tensor is contiguous.
SizeVector DefaultStrides(const SizeVector& shape);

/// 1. Separate `oldshape` into chunks of dimensions, where the dimensions are
///    ``contiguous'' in each chunk, i.e.,
///    oldstride[i] = oldshape[i+1] * oldstride[i+1]
/// 2. `newshape` must be able to be separated into same number of chunks as
///    `oldshape` was separated into, where each chunk of newshape has matching
///    ``numel'', i.e., number of subspaces, as the corresponding chunk of
///    `oldshape`.
/// Ref: aten/src/ATen/TensorUtils.cpp
std::pair<bool, SizeVector> Restride(const SizeVector& old_shape,
                                     const SizeVector& old_strides,
                                     const SizeVector& new_shape);

}  // namespace shape_util
}  // namespace core
}  // namespace open3d
