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
#include "Open3D/Container/CudaUtils.cuh"

namespace open3d {
namespace kernel {
static constexpr int MAX_DIMS = 10;
class OffsetCalculator {
public:
    OffsetCalculator(const std::vector<size_t>& src_strides,
                     const std::vector<size_t>& thread_strides)
        : num_dims_(static_cast<int>(src_strides.size())) {
#pragma unroll
        for (int i = 0; i < num_dims_; i++) {
            src_strides_[i] = static_cast<int>(src_strides[i]);
            thread_strides_[i] = static_cast<int>(thread_strides[i]);
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(int dst_idx) const {
        int src_idx = 0;
#pragma unroll
        for (int dim = 0; dim < num_dims_; dim++) {
            src_idx += dst_idx / thread_strides_[dim] * src_strides_[dim];
            dst_idx = dst_idx % thread_strides_[dim];
        }
        return src_idx;
    }

protected:
    int num_dims_;
    int src_strides_[MAX_DIMS];
    int thread_strides_[MAX_DIMS];
};

// # result.ndim == M
// # x.ndim      == N
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M],
//                            ind_2[i_1, ..., i_M],
//                            ...,
//                            ind_N[i_1, ..., i_M]]
// # result.ndim == M + 3
// # x.ndim      == N + 3
// result[A, B, i_1, ..., i_M, C] == x[A, B,
//                                     ind_1[i_1, ..., i_M],
//                                     ind_2[i_1, ..., i_M],
//                                     ...,
//                                     ind_N[i_1, ..., i_M],
//                                     C]
// E.g.
// A = np.ones((10, 20, 30, 40, 50))
// A[:, [1, 2], [2, 3], :, :]  # Supported,
//                               output_shape:  [10, 2, 40, 50]
//                               indexing_ndims:[0, 2, 2, 0, 0]
//                               slice_map:     [0, -1, 3, 4]
// A[:, [1, 2], :, [2, 3], :]  # No suport, output_shape: [2, 10, 30, 50]
//                             # For this case, a transpose op is necessary
// In our case, M == 1, since we only allow 1D indexing Tensor.
class IndexedOffsetCalculator {
public:
    IndexedOffsetCalculator(
            const std::vector<size_t>& src_strides,
            const std::vector<size_t>& src_shape,
            const std::vector<size_t>& dst_strides,
            const std::vector<size_t>& indexing_ndims,
            const std::vector<const int*>& indexing_tensor_data_ptrs) {
        src_ndims_ = src_strides.size();
        dst_ndims_ = dst_strides.size();

        bool fancy_index_visited = false;
        int size_map_next_idx = 0;
        for (int i = 0; i < src_ndims_; i++) {
            src_strides_[i] = static_cast<int>(src_strides[i]);
            src_shape_[i] = static_cast<int>(src_shape[i]);
            indexing_ndims_[i] = static_cast<int>(indexing_ndims[i]);
            indexing_tensor_data_ptrs_[i] = indexing_tensor_data_ptrs[i];

            if (indexing_ndims_[i] == 0) {
                slice_map_[size_map_next_idx] = i;
                size_map_next_idx++;
            } else if (!fancy_index_visited) {
                slice_map_[size_map_next_idx] = -1;
                size_map_next_idx++;
            }
        }

        for (int i = 0; i < dst_ndims_; i++) {
            dst_strides_[i] = static_cast<int>(dst_strides[i]);
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(size_t dst_idx) const {
        size_t src_idx = 0;
        for (size_t dim = 0; dim < dst_ndims_; dim++) {
            int64_t dim_idx = dst_idx / dst_strides_[dim];

            if (slice_map_[dim] != -1) {
                // This dim is mapped to some input slice
                src_idx += dim_idx * src_strides_[slice_map_[dim]];
            } else {
                // This dim is mapped to one or more fancy indexed input dim(s)
                for (size_t src_dim = 0; src_dim < src_ndims_; src_dim++) {
                    if (indexing_ndims_[src_dim] != 0) {
                        src_idx +=
                                indexing_tensor_data_ptrs_[src_dim][dim_idx] *
                                src_strides_[src_dim];
                    }
                }
            }
            dst_idx = dst_idx % dst_strides_[dim];
        }
        return src_idx;
    }

protected:
    int src_ndims_;
    int dst_ndims_;
    int src_strides_[MAX_DIMS];
    int src_shape_[MAX_DIMS];
    int dst_strides_[MAX_DIMS];
    int indexing_ndims_[MAX_DIMS];
    int slice_map_[MAX_DIMS];  // -1 for if that dim is fancy indexed
    const int* indexing_tensor_data_ptrs_[MAX_DIMS];
};  // namespace kernel
}  // namespace kernel
}  // namespace open3d
