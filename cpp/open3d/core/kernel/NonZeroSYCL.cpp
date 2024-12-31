// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <numeric>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "open3d/core/Indexer.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/kernel/NonZero.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

Tensor NonZeroSYCL(const Tensor& src) {
    // Get flattened non-zero indices.
    TensorIterator src_iter(src);
    const int64_t num_elements = src.NumElements();
    auto device = src.GetDevice();
    Tensor indices = Tensor::Arange(0, num_elements, 1, core::Int64, device);
    Tensor non_zero_indices(SizeVector({num_elements}), Int64, device);
    int64_t *non_zero_indices_ptr = non_zero_indices.GetDataPtr<int64_t>(),
            *indices_ptr = indices.GetDataPtr<int64_t>();
    size_t num_non_zeros;
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src.GetDtype(), [&]() {
        auto it = std::copy_if(
                oneapi::dpl::execution::dpcpp_default, indices_ptr,
                indices_ptr + num_elements, non_zero_indices_ptr,
                [src_iter](int64_t index) {
                    auto src_ptr = static_cast<const scalar_t*>(
                            src_iter.GetPtr(index));
                    OPEN3D_ASSERT(src_ptr != nullptr && "Internal error.");
                    return *src_ptr != 0;
                });
        num_non_zeros = std::distance(non_zero_indices_ptr, it);
    });

    // Transform flattened indices to indices in each dimension.
    const auto num_dims = src.NumDims();
    SizeVector shape = src.GetShape();
    // MAX_DIMS: Maximum number of dimensions of TensorRef, defined in
    // Indexer.h.
    sycl::marray<int64_t, MAX_DIMS> shape_vec;  // device copyable
    if (shape.size() > MAX_DIMS) {
        utility::LogError("Too many dimensions: {} > MAX_DIMS={}.",
                          shape.size(), MAX_DIMS);
    }
    for (auto k = 0; k < num_dims; ++k) shape_vec[k] = shape[k];
    Tensor result({num_dims, static_cast<int64_t>(num_non_zeros)}, Int64,
                  device);
    int64_t* result_ptr = result.GetDataPtr<int64_t>();
    auto queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    queue.parallel_for(num_non_zeros, [=](int64_t i) {
             auto non_zero_index = non_zero_indices_ptr[i];
             auto this_result_ptr =
                     result_ptr + i + (num_dims - 1) * num_non_zeros;
             OPEN3D_ASSERT(this_result_ptr != nullptr && "Internal error.");
             for (auto dim = num_dims - 1; dim >= 0;
                  dim--, this_result_ptr -= num_non_zeros) {
                 *this_result_ptr = non_zero_index % shape_vec[dim];
                 non_zero_index = non_zero_index / shape_vec[dim];
             }
         }).wait_and_throw();
    return result;
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
