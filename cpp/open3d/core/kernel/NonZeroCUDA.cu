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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "open3d/core/Indexer.h"
#include "open3d/core/kernel/NonZero.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename T>
struct NonZeroFunctor {
    NonZeroFunctor() {}
    __host__ __device__ bool operator()(T value) const {
        return static_cast<float>(value) != 0.0;
    }
};

struct FlatIndexTransformFunctor {
    FlatIndexTransformFunctor(const TensorIterator& iter,
                              int64_t num_non_zeros,
                              int64_t num_dims,
                              const SizeVector& shape)
        : iter_(iter), num_non_zeros_(num_non_zeros), num_dims_(num_dims) {
        for (size_t i = 0; i < shape.size(); ++i) {
            shape_[i] = shape[i];
        }
    }

    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t) {
        int64_t i = (int64_t)thrust::get<0>(t);
        int64_t non_zero_index = (int64_t)thrust::get<1>(t);

        for (int64_t dim = num_dims_ - 1; dim >= 0; dim--) {
            *static_cast<int64_t*>(iter_.GetPtr(dim * num_non_zeros_ + i)) =
                    non_zero_index % shape_[dim];
            non_zero_index = non_zero_index / shape_[dim];
        }
    }

protected:
    TensorIterator iter_;
    int64_t num_non_zeros_;
    int64_t num_dims_;
    int64_t shape_[MAX_DIMS];
};

Tensor NonZeroCUDA(const Tensor& src) {
    Tensor src_contiguous = src.Contiguous();
    const int64_t num_elements = src_contiguous.NumElements();
    const int64_t num_bytes =
            num_elements * src_contiguous.GetDtype().ByteSize();

    thrust::counting_iterator<int64_t> index_first(0);
    thrust::counting_iterator<int64_t> index_last = index_first + num_elements;

    // Get flattened non-zero indices.
    thrust::device_vector<int64_t> non_zero_indices(num_elements);
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src.GetDtype(), [&]() {
        thrust::device_ptr<const scalar_t> src_ptr(static_cast<const scalar_t*>(
                src_contiguous.GetBlob()->GetDataPtr()));

        auto it = thrust::copy_if(index_first, index_last, src_ptr,
                                  non_zero_indices.begin(),
                                  NonZeroFunctor<scalar_t>());
        non_zero_indices.resize(thrust::distance(non_zero_indices.begin(), it));
    });

    // Transform flattened indices to indices in each dimension.
    SizeVector shape = src.GetShape();
    const int64_t num_dims = src.NumDims();
    const size_t num_non_zeros = non_zero_indices.size();

    SizeVector result_shape{num_dims, static_cast<int64_t>(num_non_zeros)};
    Tensor result(result_shape, core::Int64, src.GetDevice());
    TensorIterator result_iter(result);

    index_last = index_first + num_non_zeros;
    thrust::for_each(thrust::device,
                     thrust::make_zip_iterator(thrust::make_tuple(
                             index_first, non_zero_indices.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                             index_last, non_zero_indices.end())),
                     FlatIndexTransformFunctor(result_iter, num_non_zeros,
                                               num_dims, shape));

    return result;
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
