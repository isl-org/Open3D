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

#include <vector>

#include "Open3D/Container/Broadcast.h"
#include "Open3D/Container/CudaUtils.cuh"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

static constexpr int MAX_DIMS = 10;

/// \brief Compute offset of the target Tensor \p tar_tensor, given the offset
/// of the reference Tensor \p ref_tensor.
///
/// Offset is the number of elements (not bytes) counting from the first
/// element. For example, for a contiguous Tensor of shape (2, 3, 4), the very
/// first element in memory has offset 0, while the last element has offset 2 *
/// 3 * 4 - 1 = 23.
///
/// Example use case: for unary element-wise ops, we loop through all offsets in
/// the dst Tensor, and compute each corresponding offset in the src Tensor.
///
/// Broadcast is supported. Specifically, we broadcast tar_shape to ref_shape.
/// Therefore, given ref_offset, this finds the tar_offset.
class OffsetBroadcastCalculator {
public:
    OffsetBroadcastCalculator(const SizeVector& tar_shape,
                              const SizeVector& tar_strides,
                              const SizeVector& ref_shape,
                              const SizeVector& ref_strides) {
        // Check that tar_shape can be broadcasted to ref_shape
        if (tar_shape.size() != tar_strides.size()) {
            utility::LogError("Shape size {} and strides size {} mismatch.",
                              tar_shape.size(), tar_strides.size());
        }
        if (ref_shape.size() != ref_strides.size()) {
            utility::LogError("Shape size {} and strides size {} mismatch.",
                              ref_shape.size(), ref_strides.size());
        }
        if (!CanBeBrocastedToShape(tar_shape, ref_shape)) {
            utility::LogError("Shape {} can not be broadcasted to {}.",
                              tar_shape, ref_shape);
        }

        int tar_ndims = static_cast<int>(tar_strides.size());
        int ref_ndims = static_cast<int>(ref_strides.size());
        ndims_ = ref_ndims;

        // Fill tar_shape_ and tar_strides_.
        // For each brocasted dim, set shape to 1 and stride to 0.
        //
        // [Before]
        // tar_shape:       [ 2,  1,  3]
        // tar_strides:     [ 3,  3,  1]
        // ref_shape:   [ 2,  2,  2,  3]
        // ref_strides: [12,  6,  3,  1]
        // [After]
        // tar_shape:   [ 1,  2,  1,  3]
        // tar_strides: [ 0,  3,  0,  1] <- if shape is "1", stride set to "0"
        // ref_shape:   [ 2,  2,  2,  3]
        // ref_strides: [12,  6,  3,  1]
        for (int i = 0; i < ndims_ - tar_ndims; ++i) {
            tar_shape_[i] = 1;
            tar_strides_[i] = 0;
        }
        for (int i = 0; i < tar_ndims; ++i) {
            tar_shape_[ndims_ - tar_ndims + i] = tar_shape[i];
            if (tar_shape[i] == 1) {
                tar_strides_[ndims_ - tar_ndims + i] = 0;
            } else {
                tar_strides_[ndims_ - tar_ndims + i] = tar_strides[i];
            }
        }

        // Fill ref_shape_ and ref_strides_
        for (int i = 0; i < ndims_; ++i) {
            ref_shape_[i] = ref_shape[i];
            ref_strides_[i] = ref_strides[i];
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(int ref_offset) const {
        int tar_offset = 0;
#pragma unroll
        for (int dim = 0; dim < ndims_; dim++) {
            tar_offset += ref_offset / ref_strides_[dim] * tar_strides_[dim];
            ref_offset = ref_offset % ref_strides_[dim];
        }
        return tar_offset;
    }

    static void PrintArray(const int* array, int size) {
        std::string s;
        for (int i = 0; i < size; ++i) {
            s += std::to_string(array[i]) + ", ";
        }
        utility::LogInfo("{}", s);
    }

protected:
    int ndims_;
    int tar_shape_[MAX_DIMS];
    int tar_strides_[MAX_DIMS];
    int ref_shape_[MAX_DIMS];
    int ref_strides_[MAX_DIMS];
};

// Broadcast tar_shape to ref_shape.
// That is, given ref_offset, find tar_offset.
class IndexedBrocastOffsetCalculator {
public:
    IndexedBrocastOffsetCalculator(
            const SizeVector& tar_shape,
            const SizeVector& tar_strides,
            const SizeVector& ref_shape,
            const SizeVector& ref_strides,
            const std::vector<bool>& is_trivial_dims,
            const std::vector<const int*>& indexing_tensor_data_ptrs) {
        tar_ndims_ = tar_strides.size();
        ref_ndims_ = ref_strides.size();

        bool fancy_index_visited = false;
        int size_map_next_idx = 0;
        for (int i = 0; i < tar_ndims_; i++) {
            tar_strides_[i] = static_cast<int>(tar_strides[i]);
            tar_shape_[i] = static_cast<int>(tar_shape[i]);
            is_trivial_dims_[i] = is_trivial_dims[i];
            indexing_tensor_data_ptrs_[i] = indexing_tensor_data_ptrs[i];

            if (is_trivial_dims_[i]) {
                slice_map_[size_map_next_idx] = i;
                size_map_next_idx++;
            } else if (!fancy_index_visited) {
                slice_map_[size_map_next_idx] = -1;
                size_map_next_idx++;
            }
        }

        for (int i = 0; i < ref_ndims_; i++) {
            ref_strides_[i] = static_cast<int>(ref_strides[i]);
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(int64_t ref_offset) const {
        int64_t tar_offset = 0;
#pragma unroll
        for (int64_t dim = 0; dim < ref_ndims_; dim++) {
            int64_t dim_idx = ref_offset / ref_strides_[dim];

            if (slice_map_[dim] != -1) {
                // This dim is mapped to some input slice
                tar_offset += dim_idx * tar_strides_[slice_map_[dim]];
            } else {
                // This dim is mapped to one or more fancy indexed input dim(s)
                for (int64_t tar_dim = 0; tar_dim < tar_ndims_; tar_dim++) {
                    if (!is_trivial_dims_[tar_dim]) {
                        tar_offset +=
                                indexing_tensor_data_ptrs_[tar_dim][dim_idx] *
                                tar_strides_[tar_dim];
                    }
                }
            }
            ref_offset = ref_offset % ref_strides_[dim];
        }
        return tar_offset;
    }

protected:
    int tar_ndims_;
    int ref_ndims_;

    int tar_shape_[MAX_DIMS];
    int tar_strides_[MAX_DIMS];
    int ref_shape_[MAX_DIMS];
    int ref_strides_[MAX_DIMS];

    bool is_trivial_dims_[MAX_DIMS];
    int slice_map_[MAX_DIMS];  // -1 for if that dim is fancy indexed
    const int* indexing_tensor_data_ptrs_[MAX_DIMS];
};

/// # result.ndim == M
/// # x.ndim      == N
/// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M],
///                            ind_2[i_1, ..., i_M],
///                            ...,
///                            ind_N[i_1, ..., i_M]]
/// # result.ndim == M + 3
/// # x.ndim      == N + 3
/// result[A, B, i_1, ..., i_M, C] == x[A, B,
///                                     ind_1[i_1, ..., i_M],
///                                     ind_2[i_1, ..., i_M],
///                                     ...,
///                                     ind_N[i_1, ..., i_M],
///                                     C]
/// E.g.
/// A = np.ones((10, 20, 30, 40, 50))
/// A[:, [1, 2], [2, 3], :, :]  # Supported,
///                               output_shape:  [10, 2, 40, 50]
///                               index_tensor_sizes:[0, 2, 2, 0, 0]
///                               slice_map:     [0, -1, 3, 4]
/// A[:, [1, 2], :, [2, 3], :]  # No suport, output_shape: [2, 10, 30, 50]
///                             # For this case, a transpose op is necessary
/// In our case, M == 1, since we only allow 1D indexing Tensor.
class IndexedOffsetCalculator {
public:
    IndexedOffsetCalculator(
            const SizeVector& tar_shape,
            const SizeVector& tar_strides,
            const SizeVector& ref_strides,
            const std::vector<bool>& is_trivial_dims,
            const std::vector<const int*>& indexing_tensor_data_ptrs) {
        tar_ndims_ = tar_strides.size();
        ref_ndims_ = ref_strides.size();

        bool fancy_index_visited = false;
        int size_map_next_idx = 0;
        for (int i = 0; i < tar_ndims_; i++) {
            tar_strides_[i] = static_cast<int>(tar_strides[i]);
            tar_shape_[i] = static_cast<int>(tar_shape[i]);
            is_trivial_dims_[i] = is_trivial_dims[i];
            indexing_tensor_data_ptrs_[i] = indexing_tensor_data_ptrs[i];

            if (is_trivial_dims_[i]) {
                slice_map_[size_map_next_idx] = i;
                size_map_next_idx++;
            } else if (!fancy_index_visited) {
                slice_map_[size_map_next_idx] = -1;
                size_map_next_idx++;
            }
        }

        for (int i = 0; i < ref_ndims_; i++) {
            ref_strides_[i] = static_cast<int>(ref_strides[i]);
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(int64_t ref_offset) const {
        int64_t tar_offset = 0;
#pragma unroll
        for (int64_t dim = 0; dim < ref_ndims_; dim++) {
            int64_t dim_idx = ref_offset / ref_strides_[dim];

            if (slice_map_[dim] != -1) {
                // This dim is mapped to some input slice
                tar_offset += dim_idx * tar_strides_[slice_map_[dim]];
            } else {
                // This dim is mapped to one or more fancy indexed input dim(s)
                for (int64_t tar_dim = 0; tar_dim < tar_ndims_; tar_dim++) {
                    if (!is_trivial_dims_[tar_dim]) {
                        int64_t tar_dim_idx =
                                indexing_tensor_data_ptrs_[tar_dim][dim_idx];
                        if (tar_dim_idx < 0) {
                            tar_dim_idx += tar_shape_[tar_dim];
                        }
                        assert(tar_dim_idx >= 0 &&
                               tar_dim_idx < static_cast<int64_t>(
                                                     tar_shape_[tar_dim]));
                        tar_offset += tar_dim_idx * tar_strides_[tar_dim];
                    }
                }
            }
            ref_offset = ref_offset % ref_strides_[dim];
        }
        return tar_offset;
    }

protected:
    int tar_ndims_;
    int ref_ndims_;

    int tar_shape_[MAX_DIMS];
    int tar_strides_[MAX_DIMS];
    int ref_shape_[MAX_DIMS];
    int ref_strides_[MAX_DIMS];

    bool is_trivial_dims_[MAX_DIMS];
    int slice_map_[MAX_DIMS];  // -1 for if that dim is fancy indexed
    const int* indexing_tensor_data_ptrs_[MAX_DIMS];
};

}  // namespace kernel
}  // namespace open3d
