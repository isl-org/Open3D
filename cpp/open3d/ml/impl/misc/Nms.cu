// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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
//
// Reference:
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu
//
// Reference:
// https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/src/iou3d_kernel.cu
// 3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
// Written by Shaoshuai Shi
// All Rights Reserved 2019-2020.

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "open3d/ml/Helper.h"
#include "open3d/ml/impl/misc/Nms.h"
#include "open3d/ml/impl/misc/NmsImpl.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace ml {
namespace impl {

template <typename T>
static void SortIndices(T *values,
                        int64_t *sort_indices,
                        int64_t n,
                        bool descending = false) {
    // Cast to thrust device pointer.
    thrust::device_ptr<T> values_dptr = thrust::device_pointer_cast(values);
    thrust::device_ptr<int64_t> sort_indices_dptr =
            thrust::device_pointer_cast(sort_indices);

    // Fill sort_indices with 0, 1, ..., n-1.
    thrust::sequence(sort_indices_dptr, sort_indices_dptr + n, 0);

    // Sort values and sort_indices together.
    if (descending) {
        thrust::stable_sort_by_key(values_dptr, values_dptr + n,
                                   sort_indices_dptr, thrust::greater<T>());
    } else {
        thrust::stable_sort_by_key(values_dptr, values_dptr + n,
                                   sort_indices_dptr);
    }
}

__global__ void NmsKernel(const float *boxes,
                          const int64_t *sort_indices,
                          uint64_t *mask,
                          const int n,
                          const double nms_overlap_thresh,
                          const int num_block_cols) {
    // Row-wise block index.
    const int block_row_idx = blockIdx.y;
    // Column-wise block index.
    const int block_col_idx = blockIdx.x;

    // Local block row size.
    const int row_size =
            fminf(n - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
    // Local block col size.
    const int col_size =
            fminf(n - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

    // Fill local block_boxes by fetching the global box memory.
    // block_boxes = boxes[NBS*block_col_idx : NBS*block_col_idx+col_size, :].
    //
    // TODO: It is also possible to load the comparison target to the shared
    // memory as well.
    __shared__ float block_boxes[NMS_BLOCK_SIZE * 5];
    if (threadIdx.x < col_size) {
        float *dst = block_boxes + threadIdx.x * 5;
        const int src_idx = NMS_BLOCK_SIZE * block_col_idx + threadIdx.x;
        const float *src = boxes + sort_indices[src_idx] * 5;
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
    }
    __syncthreads();

    // Comparing src and dst. In one block, the following src and dst indices
    // are compared:
    // - src: BS * block_row_idx : BS * block_row_idx + row_size
    // - dst: BS * block_col_idx : BS * block_col_idx + col_size
    //
    // With all blocks, all src and dst indices are compared.
    //
    // Result:
    // mask[i, j] is a 64-bit integer where mask[i, j][k] (k counted from right)
    // is 1 iff box[i] overlaps with box[BS*j+k].
    if (threadIdx.x < row_size) {
        // src_idx indices the global memory.
        const int src_idx = NMS_BLOCK_SIZE * block_row_idx + threadIdx.x;
        // dst_idx indices the shared memory.
        int dst_idx = block_row_idx == block_col_idx ? threadIdx.x + 1 : 0;

        uint64_t t = 0;
        while (dst_idx < col_size) {
            if (IouBev(boxes + sort_indices[src_idx] * 5,
                       block_boxes + dst_idx * 5) > nms_overlap_thresh) {
                t |= 1ULL << dst_idx;
            }
            dst_idx++;
        }
        mask[src_idx * num_block_cols + block_col_idx] = t;
    }
}

std::vector<int64_t> NmsCUDAKernel(const float *boxes,
                                   const float *scores,
                                   int n,
                                   double nms_overlap_thresh) {
    if (n == 0) {
        return {};
    }

    // Cololum-wise number of blocks.
    const int num_block_cols = utility::DivUp(n, NMS_BLOCK_SIZE);

    // Compute sort indices.
    float *scores_copy = nullptr;
    OPEN3D_ML_CUDA_CHECK(cudaMalloc((void **)&scores_copy, n * sizeof(float)));
    OPEN3D_ML_CUDA_CHECK(cudaMemcpy(scores_copy, scores, n * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
    int64_t *sort_indices = nullptr;
    OPEN3D_ML_CUDA_CHECK(
            cudaMalloc((void **)&sort_indices, n * sizeof(int64_t)));
    SortIndices(scores_copy, sort_indices, n, true);
    OPEN3D_ML_CUDA_CHECK(cudaFree(scores_copy));

    // Allocate masks on device.
    uint64_t *mask_ptr = nullptr;
    OPEN3D_ML_CUDA_CHECK(cudaMalloc((void **)&mask_ptr,
                                    n * num_block_cols * sizeof(uint64_t)));

    // Launch kernel.
    dim3 blocks(utility::DivUp(n, NMS_BLOCK_SIZE),
                utility::DivUp(n, NMS_BLOCK_SIZE));
    dim3 threads(NMS_BLOCK_SIZE);
    NmsKernel<<<blocks, threads>>>(boxes, sort_indices, mask_ptr, n,
                                   nms_overlap_thresh, num_block_cols);

    // Copy cuda masks to cpu.
    std::vector<uint64_t> mask_vec(n * num_block_cols);
    uint64_t *mask = mask_vec.data();
    OPEN3D_ML_CUDA_CHECK(cudaMemcpy(mask_vec.data(), mask_ptr,
                                    n * num_block_cols * sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost));
    OPEN3D_ML_CUDA_CHECK(cudaFree(mask_ptr));

    // Copy sort_indices to cpu.
    std::vector<int64_t> sort_indices_cpu(n);
    OPEN3D_ML_CUDA_CHECK(cudaMemcpy(sort_indices_cpu.data(), sort_indices,
                                    n * sizeof(int64_t),
                                    cudaMemcpyDeviceToHost));

    // Write to keep_indices in CPU.
    // remv_cpu has n bits in total. If the bit is 1, the corresponding
    // box will be removed.
    // TODO: This part can be implemented in CUDA. We use the original author's
    // implementation here.
    std::vector<uint64_t> remv_cpu(num_block_cols, 0);
    std::vector<int64_t> keep_indices;
    for (int i = 0; i < n; i++) {
        int block_col_idx = i / NMS_BLOCK_SIZE;
        int inner_block_col_idx = i % NMS_BLOCK_SIZE;  // threadIdx.x

        // Querying the i-th bit in remv_cpu, counted from the right.
        // - remv_cpu[block_col_idx]: the block bitmap containing the query
        // - 1ULL << inner_block_col_idx: the one-hot bitmap to extract i
        if (!(remv_cpu[block_col_idx] & (1ULL << inner_block_col_idx))) {
            // Keep the i-th box.
            keep_indices.push_back(sort_indices_cpu[i]);

            // Any box that overlaps with the i-th box will be removed.
            uint64_t *p = mask + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }
    OPEN3D_ML_CUDA_CHECK(cudaFree(sort_indices));
    return keep_indices;
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
