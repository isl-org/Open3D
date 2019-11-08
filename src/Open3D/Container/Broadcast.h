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

#include "Open3D/Container/Dispatch.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

/// \brief Returns true if two shapes are compatible for broadcasting.
/// \param left_shape Shape of the left-hand-side Tensor.
/// \param right_shape Shape of the left-hand-side Tensor.
/// \return Returns true if \p left_shape and \p right_shape are compatible for
/// broadcasting.
static bool IsCompatibleBroadcastShape(const SizeVector& left_shape,
                                       const SizeVector& right_shape) {
    size_t left_ndim = left_shape.size();
    size_t right_ndim = right_shape.size();

    if (left_ndim == 0 || right_ndim == 0) {
        return true;
    }

    // Only need to check the last `shorter_ndim` dims
    // E.g. LHS: [100, 200, 2, 3, 4]
    //      RHS:           [2, 1, 4] <- only last 3 dims need to be checked
    // Checked from right to left
    size_t shorter_ndim = std::min(left_ndim, right_ndim);
    for (size_t ind = 0; ind < shorter_ndim; ++ind) {
        size_t left_dim = left_shape[left_ndim - 1 - ind];
        size_t right_dim = left_shape[right_ndim - 1 - ind];
        if (!(left_dim == right_dim || left_dim == 1 || right_dim == 1)) {
            return false;
        }
    }
    return true;
}

/// \brief Returns the broadcasted shape of two shapes.
/// \param left_shape Shape of the left-hand-side Tensor.
/// \param right_shape Shape of the left-hand-side Tensor.
/// \return The broadcasted shape.
static SizeVector BroadcastedShape(const SizeVector& left_shape,
                                   const SizeVector& right_shape) {
    if (!IsCompatibleBroadcastShape(left_shape, right_shape)) {
        utility::LogError("Shape {} and {} are not broadcast-compatible",
                          left_shape, right_shape);
    }
    size_t left_ndim = left_shape.size();
    size_t right_ndim = right_shape.size();
    size_t shorter_ndim = std::min(left_ndim, right_ndim);
    size_t longer_ndim = std::max(left_ndim, right_ndim);

    SizeVector broadcasted_shape(longer_ndim, 0);
    // Checked from right to left
    for (size_t ind = 0; ind < shorter_ndim; ind++) {
        broadcasted_shape[longer_ndim - 1 - ind] =
                std::max(left_shape[left_ndim - 1 - ind],
                         right_shape[right_ndim - 1 - ind]);
    }
    return broadcasted_shape;
}

// TODO: only support (1,) to (x,) broadcasting
static Tensor BroadcastToShape(const Tensor& src_tensor,
                               const SizeVector& dst_shape) {
    if (src_tensor.GetShape().size() != 1 || dst_shape.size() != 1) {
        utility::LogError("Wrong 1D broadcasting shape, {}, {}",
                          src_tensor.GetShape(), dst_shape);
    }

    if (src_tensor.GetShape()[0] == 1) {
        Tensor dst_tensor(dst_shape, src_tensor.GetDtype(),
                          src_tensor.GetDevice());
        DISPATCH_DTYPE_TO_TEMPLATE(src_tensor.GetDtype(), [&]() {
            dst_tensor.Fill(src_tensor[0].Item<scalar_t>());
        });
        return dst_tensor;
    } else if (src_tensor.GetShape() == dst_shape) {
        return src_tensor;
    } else {
        utility::LogError("Wrong 1D broadcast");
    }
}

}  // namespace open3d
