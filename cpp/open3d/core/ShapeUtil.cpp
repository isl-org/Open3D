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

#include "open3d/core/ShapeUtil.h"

#include <numeric>

#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace shape_util {

/// Expand a shape with ones in front. Returning a shape with size of ndims.
/// E.g. ExpandFrontDims({2, 3}, 5) == {1, 1, 1, 2, 3}
static SizeVector ExpandFrontDims(const SizeVector& shape, int64_t ndims) {
    if (ndims < static_cast<int64_t>(shape.size())) {
        utility::LogError("Cannot expand a shape with ndims {} to ndims {}.",
                          shape.size(), ndims);
    }
    SizeVector expanded_shape(ndims, 1);
    std::copy(shape.begin(), shape.end(),
              expanded_shape.begin() + ndims - shape.size());
    return expanded_shape;
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

int64_t WrapDim(int64_t dim, int64_t max_dim, bool inclusive) {
    if (max_dim <= 0) {
        utility::LogError("max_dim {} must be >= 0");
    }
    int64_t min = -max_dim;
    int64_t max = inclusive ? max_dim : max_dim - 1;

    if (dim < min || dim > max) {
        utility::LogError(
                "Index out-of-range: dim == {}, but it must satisfy {} <= dim "
                "<= {}",
                dim, min, max);
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
    int64_t inferred_dim = 0;
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

SizeVector Concat(const SizeVector& l_shape, const SizeVector& r_shape) {
    SizeVector dst_shape = l_shape;
    dst_shape.insert(dst_shape.end(), r_shape.begin(), r_shape.end());
    return dst_shape;
}

SizeVector Iota(int64_t n) {
    if (n < 0) {
        utility::LogError("Iota(n) requires n >= 0, but n == {}.", n);
    }
    SizeVector sv(n);
    std::iota(sv.begin(), sv.end(), 0);
    return sv;
}

SizeVector DefaultStrides(const SizeVector& shape) {
    SizeVector strides(shape.size());
    int64_t stride_size = 1;
    for (int64_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride_size;
        // Handles 0-sized dimensions
        stride_size *= std::max<int64_t>(shape[i - 1], 1);
    }
    return strides;
}

std::pair<bool, SizeVector> Restride(const SizeVector& old_shape,
                                     const SizeVector& old_strides,
                                     const SizeVector& new_shape) {
    if (old_shape.empty()) {
        return std::make_pair(true, SizeVector(new_shape.size(), 1));
    }

    // NOTE: Stride is arbitrary in the numel() == 0 case. To match NumPy
    // behavior we copy the strides if the size matches, otherwise we use the
    // stride as if it were computed via resize. This could perhaps be combined
    // with the below code, but the complexity didn't seem worth it.
    int64_t numel = old_shape.NumElements();
    if (numel == 0 && old_shape == new_shape) {
        return std::make_pair(true, old_strides);
    }

    SizeVector new_strides(new_shape.size());
    if (numel == 0) {
        for (int64_t view_d = new_shape.size() - 1; view_d >= 0; view_d--) {
            if (view_d == (int64_t)(new_shape.size() - 1)) {
                new_strides[view_d] = 1;
            } else {
                new_strides[view_d] =
                        std::max<int64_t>(new_shape[view_d + 1], 1) *
                        new_strides[view_d + 1];
            }
        }
        return std::make_pair(true, new_strides);
    }

    int64_t view_d = new_shape.size() - 1;
    // Stride for each subspace in the chunk
    int64_t chunk_base_stride = old_strides.back();
    // Numel in current chunk
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = old_shape.size() - 1; tensor_d >= 0; tensor_d--) {
        tensor_numel *= old_shape[tensor_d];
        // If end of tensor size chunk, check view
        if ((tensor_d == 0) ||
            (old_shape[tensor_d - 1] != 1 &&
             old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
            while (view_d >= 0 &&
                   (view_numel < tensor_numel || new_shape[view_d] == 1)) {
                new_strides[view_d] = view_numel * chunk_base_stride;
                view_numel *= new_shape[view_d];
                view_d--;
            }
            if (view_numel != tensor_numel) {
                return std::make_pair(false, SizeVector());
            }
            if (tensor_d > 0) {
                chunk_base_stride = old_strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if (view_d != -1) {
        return std::make_pair(false, SizeVector());
    }
    return std::make_pair(true, new_strides);
}

}  // namespace shape_util
}  // namespace core
}  // namespace open3d
