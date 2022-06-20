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

#include <numeric>

#include "open3d/core/Indexer.h"
#include "open3d/core/kernel/NonZero.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor NonZeroCPU(const Tensor& src) {
    // Get flattened non-zero indices.
    TensorIterator src_iter(src);
    const int64_t num_elements = src.NumElements();
    std::vector<int64_t> indices(static_cast<size_t>(num_elements));
    std::iota(std::begin(indices), std::end(indices), 0);
    std::vector<int64_t> non_zero_indices(num_elements);
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src.GetDtype(), [&]() {
        auto it = std::copy_if(
                indices.begin(), indices.end(), non_zero_indices.begin(),
                [&src_iter](int64_t index) {
                    const void* src_ptr = src_iter.GetPtr(index);
                    OPEN3D_ASSERT(src_ptr != nullptr && "Internal error.");
                    return static_cast<float>(
                                   *static_cast<const scalar_t*>(src_ptr)) != 0;
                });
        non_zero_indices.resize(std::distance(non_zero_indices.begin(), it));
    });

    // Transform flattened indices to indices in each dimension.
    SizeVector shape = src.GetShape();
    const int64_t num_dims = src.NumDims();
    const size_t num_non_zeros = non_zero_indices.size();

    SizeVector result_shape{num_dims, static_cast<int64_t>(num_non_zeros)};
    Tensor result(result_shape, core::Int64, src.GetDevice());
    TensorIterator result_iter(result);

    std::vector<std::vector<int64_t>> non_zero_indices_by_dimensions(
            num_dims, std::vector<int64_t>(num_non_zeros, 0));
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < static_cast<int64_t>(num_non_zeros); i++) {
        int64_t non_zero_index = non_zero_indices[i];
        for (int64_t dim = num_dims - 1; dim >= 0; dim--) {
            void* result_ptr = result_iter.GetPtr(dim * num_non_zeros + i);
            OPEN3D_ASSERT(result_ptr != nullptr && "Internal error.");
            *static_cast<int64_t*>(result_ptr) = non_zero_index % shape[dim];
            non_zero_index = non_zero_index / shape[dim];
        }
    }

    return result;
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
