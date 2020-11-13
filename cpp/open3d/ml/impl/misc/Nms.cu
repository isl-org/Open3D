/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "open3d/ml/impl/misc/Nms.h"
#include "open3d/ml/impl/misc/NmsImpl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

namespace open3d {
namespace ml {
namespace impl {

static void SortIndices(float *values,
                        int64_t *sort_indices,
                        int64_t N,
                        bool descending = false) {
    // Cast to thrust device pointer.
    thrust::device_ptr<float> values_dptr = thrust::device_pointer_cast(values);
    thrust::device_ptr<int64_t> sort_indices_dptr =
            thrust::device_pointer_cast(sort_indices);

    // Fill sort_indices with 0, 1, ..., N-1.
    thrust::sequence(sort_indices_dptr, sort_indices_dptr + N, 0);

    // Sort values and sort_indices together.
    if (descending) {
        thrust::stable_sort_by_key(values_dptr, values_dptr + N,
                                   sort_indices_dptr, thrust::greater<float>());
    } else {
        thrust::stable_sort_by_key(values_dptr, values_dptr + N,
                                   sort_indices_dptr);
    }
}

__global__ void nms_kernel(const float *boxes,
                           const int64_t *sort_indices,
                           uint64_t *mask,
                           const int N,
                           const double nms_overlap_thresh) {
    // boxes: (N, 5)
    // mask:  (N, N/BS)
    //
    // Kernel launch
    // blocks : (N/BS, N/BS)
    // threads: BS

    // Row-wise block index.
    const int block_row_idx = blockIdx.y;
    // Column-wise block index.
    const int block_col_idx = blockIdx.x;

    // Local block row size.
    const int row_size =
            fminf(N - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
    // Local block col size.
    const int col_size =
            fminf(N - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

    // Cololum-wise number of blocks.
    const int num_block_cols = DIVUP(N, NMS_BLOCK_SIZE);

    // Fill local block_boxes by fetching the global box memory.
    // block_boxes = boxes[NBS*block_col_idx : NBS*block_col_idx+col_size, :].
    //
    // TODO: It is also possible to load the comparison target to the shared
    // memory as well.
    __shared__ float block_boxes[NMS_BLOCK_SIZE * 5];
    if (threadIdx.x < col_size) {
        float *dst = block_boxes + threadIdx.x * 5;
        const int src_idx = NMS_BLOCK_SIZE * block_row_idx + threadIdx.x;
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
            if (iou_bev(boxes + sort_indices[src_idx] * 5,
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
                                   int N,
                                   double nms_overlap_thresh) {
    const int num_block_cols = DIVUP(N, NMS_BLOCK_SIZE);

    // Compute sort indices.
    float *scores_copy = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&scores_copy, N * sizeof(float)));
    CHECK_ERROR(cudaMemcpy(scores_copy, scores, N * sizeof(float),
                           cudaMemcpyDeviceToDevice));
    int64_t *sort_indices = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&sort_indices, N * sizeof(int64_t)));
    SortIndices(scores_copy, sort_indices, N, true);

    // Allocate masks on device.
    uint64_t *mask_ptr = nullptr;
    CHECK_ERROR(cudaMalloc((void **)&mask_ptr,
                           N * num_block_cols * sizeof(uint64_t)));

    // Launch kernel.
    dim3 blocks(DIVUP(N, NMS_BLOCK_SIZE), DIVUP(N, NMS_BLOCK_SIZE));
    dim3 threads(NMS_BLOCK_SIZE);
    nms_kernel<<<blocks, threads>>>(boxes, sort_indices, mask_ptr, N,
                                    nms_overlap_thresh);

    // Copy cuda masks to cpu.
    std::vector<uint64_t> mask_vec(N * num_block_cols);
    uint64_t *mask = mask_vec.data();
    CHECK_ERROR(cudaMemcpy(mask_vec.data(), mask_ptr,
                           N * num_block_cols * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaFree(mask_ptr));

    // Copy sort_indices to cpu.
    std::vector<int64_t> sort_indices_cpu(N);
    CHECK_ERROR(cudaMemcpy(sort_indices_cpu.data(), sort_indices,
                           N * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Write to keep_indices in CPU.
    // remv_cpu has N bits in total. If the bit is 1, the corresponding
    // box will be removed.
    // TODO: This part can be implemented in CUDA. We use the original author's
    // implementation here.
    std::vector<uint64_t> remv_cpu(num_block_cols, 0);
    std::vector<int64_t> keep_indices;
    for (int i = 0; i < N; i++) {
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
    CHECK_ERROR(cudaFree(sort_indices));
    return keep_indices;
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
