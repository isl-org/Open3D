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

#include "Open3D/Core/ShapeUtil.h"

#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {
namespace shape_util {

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

SizeVector ReductionShape(const SizeVector& src_shape,
                          const SizeVector& dims,
                          bool keepdim) {
    int64_t src_ndims = src_shape.size();
    SizeVector out_shape = src_shape;

    // WrapDim throws exception if out-of-range.
    if (keepdim) {
        for (const int64_t& dim : dims) {
            out_shape[WrapDim(dim, src_ndims)] = 1;
        }
    } else {
        // If dim i is reduced, dims_mask[i] == true.
        std::vector<bool> dims_mask(src_ndims, false);
        for (const int64_t& dim : dims) {
            if (dims_mask[WrapDim(dim, src_ndims)]) {
                utility::LogError("Repeated reduction dimension {}", dim);
            }
            dims_mask[WrapDim(dim, src_ndims)] = true;
        }
        int64_t to_fill = 0;
        for (int64_t i = 0; i < src_ndims; ++i) {
            if (!dims_mask[i]) {
                out_shape[to_fill] = out_shape[i];
                to_fill++;
            }
        }
        out_shape.resize(to_fill);
    }
    return out_shape;
}

int64_t WrapDim(int64_t dim, int64_t max_dim) {
    if (max_dim <= 0) {
        utility::LogError("max_dim {} must be >= 0");
    }
    if (dim < -max_dim || dim > max_dim - 1) {
        utility::LogError(
                "Index out-of-range: dim == {}, but it must satisfy {} <= dim "
                "<= {}",
                dim, 0, max_dim - 1);
    }
    if (dim < 0) {
        dim += max_dim;
    }
    return dim;
}

SizeVector InferShape(SizeVector shape, int64_t num_elements) {
    SizeVector inferred_shape = shape;
    int64_t new_size = 1;
    bool has_inferred_dim = false;
    int64_t inferred_dim;
    for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
        if (shape[dim] == -1) {
            if (has_inferred_dim) {
                utility::LogError(
                        "Proposed shape {}, but at most one dimension can be "
                        "-1 (inferred).",
                        shape.ToString());
            }
            inferred_dim = dim;
            has_inferred_dim = true;
        } else if (shape[dim] >= 0) {
            new_size *= shape[dim];
        } else {
            utility::LogError("Invalid shape dimension {}", shape[dim]);
        }
    }

    if (num_elements == new_size ||
        (has_inferred_dim && new_size > 0 && num_elements % new_size == 0)) {
        if (has_inferred_dim) {
            // We have a degree of freedom here to select the dimension size;
            // follow NumPy semantics and just bail. However, a nice error
            // message is needed because users often use `view` as a way to
            // flatten & unflatten dimensions and will otherwise be confused why
            //   empty_tensor.view( 0, 0)
            // works yet
            //   empty_tensor.view(-1, 0)
            // doesn't.
            if (new_size == 0) {
                utility::LogError(
                        "Cannot reshape tensor of 0 elements into shape {}, "
                        "because the unspecified dimension size -1 can be any "
                        "value and is ambiguous.",
                        shape.ToString());
            }
            inferred_shape[inferred_dim] = num_elements / new_size;
        }
        return inferred_shape;
    }

    utility::LogError("Shape {} is invalid for {} number of elements.", shape,
                      num_elements);
}

}  // namespace shape_util
}  // namespace open3d
