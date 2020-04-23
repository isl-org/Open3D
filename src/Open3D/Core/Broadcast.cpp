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

#include "Open3D/Core/Broadcast.h"

#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {

/// Expand a shape with ones in front. Returning a shape with size of ndims.
/// E.g. ExpandFrontDims({2, 3}, 5) == {1, 1, 1, 2, 3}
const SizeVector ExpandFrontDims(const SizeVector& shape, int64_t ndims) {
    if (ndims < static_cast<int64_t>(shape.size())) {
        utility::LogError("Cannot expand a shape with ndims {} to ndims {}.",
                          shape.size(), ndims);
    }
    SizeVector expanded_shape(ndims, 1);
    std::copy(shape.begin(), shape.end(),
              expanded_shape.begin() + ndims - shape.size());
    return std::move(expanded_shape);
}

bool IsCompatibleBroadcastShape(const SizeVector& l_shape,
                                const SizeVector& r_shape) {
    int64_t l_ndims = l_shape.size();
    int64_t r_ndims = r_shape.size();

    if (l_ndims == 0 || r_ndims == 0) {
        return true;
    }

    // Only need to check the last `shorter_ndims` dims
    // E.g. LHS: [100, 200, 2, 3, 4]
    //      RHS:           [2, 1, 4] <- only last 3 dims need to be checked
    // Checked from right to left
    int64_t shorter_ndims = std::min(l_ndims, r_ndims);
    for (int64_t i = 0; i < shorter_ndims; ++i) {
        int64_t l_dim = l_shape[l_ndims - 1 - i];
        int64_t r_dim = r_shape[r_ndims - 1 - i];
        if (!(l_dim == r_dim || l_dim == 1 || r_dim == 1)) {
            return false;
        }
    }
    return true;
}

SizeVector BroadcastedShape(const SizeVector& l_shape,
                            const SizeVector& r_shape) {
    if (!IsCompatibleBroadcastShape(l_shape, r_shape)) {
        utility::LogError("Shape {} and {} are not broadcast-compatible",
                          l_shape, r_shape);
    }

    int64_t l_ndims = l_shape.size();
    int64_t r_ndims = r_shape.size();
    int64_t out_ndims = std::max(l_ndims, r_ndims);

    // Fill omitted dimensions with shape 1.
    SizeVector l_shape_filled = ExpandFrontDims(l_shape, out_ndims);
    SizeVector r_shape_filled = ExpandFrontDims(r_shape, out_ndims);

    SizeVector broadcasted_shape(out_ndims);
    for (int64_t i = 0; i < out_ndims; i++) {
        if (l_shape_filled[i] == 1) {
            broadcasted_shape[i] = r_shape_filled[i];
        } else if (r_shape_filled[i] == 1) {
            broadcasted_shape[i] = l_shape_filled[i];
        } else if (l_shape_filled[i] == r_shape_filled[i]) {
            broadcasted_shape[i] = l_shape_filled[i];
        } else {
            utility::LogError(
                    "Internal error: dimension size {} is not compatible with "
                    "{}, however, this error shall have been captured by "
                    "IsCompatibleBroadcastShape already.",
                    l_shape_filled[i], r_shape_filled[i]);
        }
    }
    return broadcasted_shape;
}

bool CanBeBrocastedToShape(const SizeVector& src_shape,
                           const SizeVector& dst_shape) {
    if (IsCompatibleBroadcastShape(src_shape, dst_shape)) {
        return BroadcastedShape(src_shape, dst_shape) == dst_shape;
    } else {
        return false;
    }
}

}  // namespace open3d
