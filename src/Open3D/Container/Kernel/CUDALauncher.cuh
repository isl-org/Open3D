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

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Container/CudaUtils.cuh"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

static constexpr int threads_per_block = 128;
static constexpr int items_per_thread = 4;
static constexpr int MAX_DIMS = 10;

namespace open3d {
namespace kernel {

// Applies f for each element
// Works for unary / binary elementwise operations
template <int threads_per_block, int items_per_thread, typename func_t>
__global__ void ElementWiseKernel(int N, func_t f) {
    int items_per_block = threads_per_block * items_per_thread;
    int idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        if (idx < N) {
            f(idx);
            idx += threads_per_block;
        }
    }
}

class CUDALauncher {
public:
    /// Recover src tensor element offsets given dst tensor element offsets
    /// src and dst tensors have the same size but different strides
    class OffsetCalculator {
    public:
        OffsetCalculator(size_t num_dims,
                         const size_t* src_strides,
                         const size_t* dst_strides)
            : num_dims_(static_cast<int>(num_dims)) {
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
                const std::vector<size_t>& dst_strides,
                const std::vector<size_t>& indexing_shapes,
                const std::vector<const int*>& indexing_tensor_data_ptrs)
            : num_dims_(src_strides.size()) {
            for (int i = 0; i < num_dims_; i++) {
                src_strides_[i] = static_cast<int>(src_strides[i]);
                dst_strides_[i] = static_cast<int>(dst_strides[i]);
                indexing_shapes_[i] = static_cast<int>(indexing_shapes[i]);
                indexing_tensor_data_ptrs_[i] = indexing_tensor_data_ptrs[i];
            }
        }

        OPEN3D_HOST_DEVICE int GetOffset(size_t thread_idx) const {
            size_t output_idx = 0;
            for (size_t dim = 0; dim < num_dims_; dim++) {
                size_t dim_idx = thread_idx / dst_strides_[dim];
                size_t dim_size = indexing_shapes_[dim];

                if (dim_size == 0) {
                    output_idx += dim_idx * src_strides_[dim];
                } else if (dim_size == 1) {
                    // TODO assert, negative indexing; reduce dimension
                    output_idx += indexing_tensor_data_ptrs_[dim][0] *
                                  src_strides_[dim];
                } else {
                    // TODO assert, negative indexing
                    output_idx += indexing_tensor_data_ptrs_[dim][dim_idx] *
                                  src_strides_[dim];
                }
                thread_idx = thread_idx % dst_strides_[dim];
            }
            return output_idx;
        }

    protected:
        int num_dims_;
        int src_strides_[MAX_DIMS];
        int dst_strides_[MAX_DIMS];
        int indexing_shapes_[MAX_DIMS];
        const int* indexing_tensor_data_ptrs_[MAX_DIMS];
    };

public:
    template <typename scalar_t, typename func_t>
    static void LaunchUnaryEWKernel(const Tensor& src,
                                    Tensor& dst,
                                    func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
        int N = static_cast<int>(src.GetShape().NumElements());
        int items_per_block = threads_per_block * items_per_thread;
        int grid_size = (N + items_per_block - 1) / items_per_block;

        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        int element_byte_size = DtypeUtil::ByteSize(src.GetDtype());
        OffsetCalculator offset_calculator(src.GetShape().size(),
                                           src.GetStrides().data(),
                                           dst.GetStrides().data());

        auto f = [=] OPEN3D_HOST_DEVICE(int dst_idx) {
            int src_idx = offset_calculator.GetOffset(dst_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        };

        ElementWiseKernel<threads_per_block, items_per_thread>
                <<<grid_size, threads_per_block, 0>>>(N, f);
    }

    template <typename scalar_t, typename func_t>
    static void LaunchIndexedUnaryEWKernel(const Tensor& src,
                                           Tensor& dst,
                                           const std::vector<Tensor>& indices,
                                           const SizeVector& indexing_shapes,
                                           func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
        int N = static_cast<int>(dst.GetShape().NumElements());
        int items_per_block = threads_per_block * items_per_thread;
        int grid_size = (N + items_per_block - 1) / items_per_block;

        std::vector<const int*> indexing_tensor_data_ptrs;
        for (int i = 0; i < indices.size(); ++i) {
            auto index_tensor_ptr =
                    static_cast<const int*>(indices[i].GetDataPtr());
            indexing_tensor_data_ptrs.push_back(index_tensor_ptr);
        }

        IndexedOffsetCalculator src_offset_calculator(
                src.GetStrides(), dst.GetStrides(), indexing_shapes,
                indexing_tensor_data_ptrs);

        const char* src_data_ptr = static_cast<const char*>(src.GetDataPtr());
        char* dst_data_ptr = static_cast<char*>(dst.GetDataPtr());
        int element_byte_size = DtypeUtil::ByteSize(src.GetDtype());

        auto f = [=] OPEN3D_HOST_DEVICE(int dst_idx) {
            int src_idx = src_offset_calculator.GetOffset(dst_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            void* dst_ptr = dst_data_ptr + dst_idx * element_byte_size;
            element_kernel(src_ptr, dst_ptr);
        };

        ElementWiseKernel<threads_per_block, items_per_thread>
                <<<grid_size, threads_per_block, 0>>>(N, f);
    }
};

}  // namespace kernel
}  // namespace open3d
