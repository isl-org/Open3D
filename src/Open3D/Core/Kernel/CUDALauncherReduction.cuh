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

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

static constexpr int64_t default_grid_size = 64;
static constexpr int64_t default_block_size = 256;

namespace cg = cooperative_groups;

namespace open3d {
namespace kernel {
namespace cuda_launcher {

template <typename T>
struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// Specialize for double to avoid unaligned memory access compile errors.
template <>
struct SharedMemory<double> {
    __device__ inline operator double*() {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

// The kernel needs a minimum of 64 * sizeof(scalar_t) bytes of shared
// memory.
// - blockDim.x <= 32: allocate 64 * sizeof(scalar_t) bytes.
// - blockDim.x > 32 : allocate blockDim.x * sizeof(scalar_t) bytes.
template <typename scalar_t>
int64_t GetSMSize(int64_t grid_size, int64_t block_size) {
    return (block_size <= 32) ? 2 * block_size * sizeof(scalar_t)
                              : block_size * sizeof(scalar_t);
}

std::pair<int64_t, int64_t> GetGridSizeBlockSize(int64_t n) {
    static auto NextPow2 = [](int64_t x) -> int64_t {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    };

    // block_size = NextPow2(ceil(n / 2))
    int64_t block_size = NextPow2((n + 1) / 2);
    block_size = std::min(block_size, default_block_size);

    // grid_size = ceil(n / (block_size * 2))
    int64_t grid_size = (n + (block_size * 2 - 1)) / (block_size * 2);
    grid_size = std::min(grid_size, default_grid_size);

    return std::make_pair(grid_size, block_size);
}

template <typename scalar_t, typename func_t>
__global__ void ReduceKernelInit(Indexer indexer,
                                 scalar_t identity,
                                 func_t element_kernel,
                                 scalar_t* g_odata,
                                 unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    scalar_t* sdata = SharedMemory<scalar_t>();
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int grid_stride = blockDim.x * 2 * gridDim.x;
    unsigned int output_idx = blockIdx.y;

    // Reduce multiple elements per thread. Larger gridDim.x results in larger
    // grid_stride and fewer elements per thread.
    scalar_t local_result = identity;
    while (i < n) {
        // local_result += g_idata[i];
        element_kernel(indexer.GetReductionInputPtr(output_idx, i),
                       &local_result);
        if (i + blockDim.x < n) {
            // local_result += g_idata[i + blockDim.x];
            element_kernel(
                    indexer.GetReductionInputPtr(output_idx, i + blockDim.x),
                    &local_result);
        }
        i += grid_stride;
    }
    sdata[tid] = local_result;
    cg::sync(cta);

    // Unrolled: 512, 256, 128.
    if (blockDim.x >= 512 && tid < 256) {
        // local_result += sdata[tid + 256];
        element_kernel(&sdata[tid + 256], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 256 && tid < 128) {
        // local_result += sdata[tid + 128];
        element_kernel(&sdata[tid + 128], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 128 && tid < 64) {
        // local_result += sdata[tid + 64];
        element_kernel(&sdata[tid + 64], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);

    // Single warp reduction with shuffle: 64, 32, 16, 8, 4, 2, 1
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    scalar_t local_temp = identity;
    if (cta.thread_rank() < 32) {
        // Fetch final intermediate result from 2nd warp
        if (blockDim.x >= 64) {
            // local_result += sdata[tid + 32];
            element_kernel(&sdata[tid + 32], &local_result);
        }
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            if (blockDim.x >= offset * 2) {
                local_temp = tile32.shfl_down(local_result, offset);
                element_kernel(&local_temp, &local_result);
            }
        }
    }

    // Write result for this block to global mem.
    if (cta.thread_rank() == 0) {
        g_odata[blockIdx.y * gridDim.x + blockIdx.x] = local_result;
    }
}

template <typename scalar_t, typename func_t>
__global__ void ReduceKernelBlock(scalar_t identity,
                                  func_t element_kernel,
                                  scalar_t* g_idata,
                                  scalar_t* g_odata,
                                  unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    scalar_t* sdata = SharedMemory<scalar_t>();
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int grid_stride = blockDim.x * 2 * gridDim.x;
    unsigned int output_idx = blockIdx.y;

    // Reduce multiple elements per thread. Larger gridDim.x results in larger
    // grid_stride and fewer elements per thread.
    scalar_t local_result = identity;
    while (i < n) {
        // local_result += g_idata[i];
        element_kernel(&g_idata[blockIdx.y * gridDim.x + i], &local_result);
        if (i + blockDim.x < n) {
            // local_result += g_idata[i + blockDim.x];
            element_kernel(&g_idata[blockIdx.y * gridDim.x + i + blockDim.x],
                           &local_result);
        }
        i += grid_stride;
    }
    sdata[tid] = local_result;
    cg::sync(cta);

    // Unrolled: 512, 256, 128.
    if (blockDim.x >= 512 && tid < 256) {
        // local_result += sdata[tid + 256];
        element_kernel(&sdata[tid + 256], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 256 && tid < 128) {
        // local_result += sdata[tid + 128];
        element_kernel(&sdata[tid + 128], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);
    if (blockDim.x >= 128 && tid < 64) {
        // local_result += sdata[tid + 64];
        element_kernel(&sdata[tid + 64], &local_result);
        sdata[tid] = local_result;
    }
    cg::sync(cta);

    // Single warp reduction with shuffle: 64, 32, 16, 8, 4, 2, 1
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    scalar_t local_temp = identity;
    if (cta.thread_rank() < 32) {
        // Fetch final intermediate result from 2nd warp
        if (blockDim.x >= 64) {
            // local_result += sdata[tid + 32];
            element_kernel(&sdata[tid + 32], &local_result);
        }
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            if (blockDim.x >= offset * 2) {
                local_temp = tile32.shfl_down(local_result, offset);
                element_kernel(&local_temp, &local_result);
            }
        }
    }

    // Write result for this block to global mem.
    if (cta.thread_rank() == 0) {
        g_odata[blockIdx.y * gridDim.x + blockIdx.x] = local_result;
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionKernel(const Indexer& indexer,
                           scalar_t identity,
                           func_t element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

    int64_t num_inputs = indexer.NumWorkloads();
    int64_t num_outputs = indexer.NumOutputElements();
    int64_t ipo = num_inputs / num_outputs;  // Inputs per output

    int64_t grid_size_x = 0;
    int64_t grid_size_y = num_outputs;
    int64_t block_size = 0;
    std::tie(grid_size_x, block_size) = GetGridSizeBlockSize(ipo);

    dim3 grid_dim(grid_size_x, grid_size_y, 1);
    dim3 block_dim(block_size, 1, 1);

    int64_t total_grid_size = grid_size_x * grid_size_y;

    // Allocate device temporary memory. d_odata and d_tdata are double buffers
    // for recursive reductions.
    scalar_t* d_odata = nullptr;  // Device output, grid_size elements
    scalar_t* d_tdata = nullptr;  // Device temp output, grid_size elements
    OPEN3D_CUDA_CHECK(
            cudaMalloc((void**)&d_odata, total_grid_size * sizeof(scalar_t)));
    OPEN3D_CUDA_CHECK(
            cudaMalloc((void**)&d_tdata, total_grid_size * sizeof(scalar_t)));

    // First pass reduction, read from Tensor.
    utility::LogDebug("ipo={}, grid_dim.x={}, grid_dim.y={}, block_size={}",
                      ipo, grid_dim.x, grid_dim.y, block_size);
    ReduceKernelInit<scalar_t>
            <<<grid_dim, block_dim,
               GetSMSize<scalar_t>(grid_size_x, block_size)>>>(
                    indexer, identity, element_kernel, d_odata, ipo);
    OPEN3D_GET_LAST_CUDA_ERROR("Kernel execution failed.");

    // Reduce the partial results from blocks. No need to read from Tensor.
    ipo = grid_size_x;
    while (ipo > 1) {
        std::tie(grid_size_x, block_size) = GetGridSizeBlockSize(ipo);
        grid_dim = dim3(grid_size_x, num_outputs, 1);
        block_dim = dim3(block_size, 1, 1);
        utility::LogDebug("ipo={}, grid_dim.x={}, grid_dim.y={}, block_size={}",
                          ipo, grid_dim.x, grid_dim.y, block_size);
        // Input: d_tdata, output: d_odata
        OPEN3D_CUDA_CHECK(cudaMemcpy(d_tdata, d_odata,
                                     ipo * grid_size_y * sizeof(scalar_t),
                                     cudaMemcpyDeviceToDevice));
        ReduceKernelBlock<scalar_t>
                <<<grid_dim, block_dim,
                   GetSMSize<scalar_t>(grid_size_x, block_size)>>>(
                        identity, element_kernel, d_tdata, d_odata, ipo);
        OPEN3D_GET_LAST_CUDA_ERROR("Kernel execution failed.");
        ipo = (ipo + (block_size * 2 - 1)) / (block_size * 2);
    }

    OPEN3D_CUDA_CHECK(cudaMemcpy(indexer.GetOutputPtr(0), d_odata,
                                 grid_size_y * sizeof(scalar_t),
                                 cudaMemcpyDeviceToHost));

    // Clean up
    OPEN3D_CUDA_CHECK(cudaFree(d_odata));
    OPEN3D_CUDA_CHECK(cudaFree(d_tdata));
}

}  // namespace cuda_launcher
}  // namespace kernel
}  // namespace open3d
