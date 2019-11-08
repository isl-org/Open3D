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

#include "Open3D/Container/Kernel/Scheduler.h"
#include "Open3D/Container/CudaUtils.cuh"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

static constexpr int threads_per_block = 128;
static constexpr int items_per_thread = 4;

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

        SizeVector default_strides = Tensor::DefaultStrides(src.GetShape());
        OffsetCalculator src_offset_calculator(src.GetStrides(),
                                               default_strides);
        OffsetCalculator dst_offset_calculator(dst.GetStrides(),
                                               default_strides);

        auto f = [=] OPEN3D_HOST_DEVICE(int dst_raw_idx) {
            int src_idx = src_offset_calculator.GetOffset(dst_raw_idx);
            const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
            int dst_idx = dst_offset_calculator.GetOffset(dst_raw_idx);
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
                src.GetStrides(), src.GetShape(), dst.GetStrides(),
                indexing_shapes, indexing_tensor_data_ptrs);

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
