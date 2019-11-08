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
                     const std::vector<size_t>& dst_strides)
        : num_dims_(static_cast<int>(src_strides.size())) {
#pragma unroll
        for (int i = 0; i < num_dims_; i++) {
            src_strides_[i] = static_cast<int>(src_strides[i]);
            dst_strides_[i] = static_cast<int>(dst_strides[i]);
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(int dst_idx) const {
        int src_idx = 0;
#pragma unroll
        for (int dim = 0; dim < num_dims_; dim++) {
            src_idx += dst_idx / dst_strides_[dim] * src_strides_[dim];
            dst_idx = dst_idx % dst_strides_[dim];
        }
        return src_idx;
    }

protected:
    int num_dims_;
    int src_strides_[MAX_DIMS];
    int dst_strides_[MAX_DIMS];
};

class IndexedOffsetCalculator {
public:
    IndexedOffsetCalculator(
            const std::vector<size_t>& src_strides,
            const std::vector<size_t>& src_shape,
            const std::vector<size_t>& dst_strides,
            const std::vector<size_t>& indexing_shapes,
            const std::vector<const int*>& indexing_tensor_data_ptrs)
        : num_dims_(src_strides.size()) {
        for (int i = 0; i < num_dims_; i++) {
            src_strides_[i] = static_cast<int>(src_strides[i]);
            src_shape_[i] = static_cast<int>(src_shape[i]);
            dst_strides_[i] = static_cast<int>(dst_strides[i]);
            indexing_shapes_[i] = static_cast<int>(indexing_shapes[i]);
            indexing_tensor_data_ptrs_[i] = indexing_tensor_data_ptrs[i];
        }
    }

    OPEN3D_HOST_DEVICE int GetOffset(size_t thread_idx) const {
        size_t output_idx = 0;
        for (size_t dim = 0; dim < num_dims_; dim++) {
            int64_t dim_idx = thread_idx / dst_strides_[dim];
            size_t dim_size = indexing_shapes_[dim];

            // clang-format off
                dim_idx = (dim_size == 0) ? dim_idx
                  : ((dim_size == 1)
                     ? indexing_tensor_data_ptrs_[dim][0]
                     : indexing_tensor_data_ptrs_[dim][dim_idx]);
            // clang-format on

            assert(dim_idx >= -(int64_t)src_shape_[dim] &&
                   dim_idx < (int64_t)src_shape_[dim]);
            dim_idx = (dim_idx >= 0) ? dim_idx : src_shape_[dim] + dim_idx;

            output_idx += dim_idx * src_strides_[dim];
            thread_idx = thread_idx % dst_strides_[dim];
        }
        return output_idx;
    }

protected:
    int num_dims_;
    int src_strides_[MAX_DIMS];
    int src_shape_[MAX_DIMS];
    int dst_strides_[MAX_DIMS];
    int indexing_shapes_[MAX_DIMS];
    const int* indexing_tensor_data_ptrs_[MAX_DIMS];
};
}  // namespace kernel
}  // namespace open3d
