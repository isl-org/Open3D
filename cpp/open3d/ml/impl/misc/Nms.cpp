#include "open3d/ml/impl/misc/Nms.h"

#include <iostream>
#include <numeric>

#include "open3d/ml/impl/misc/NmsImpl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace open3d {
namespace ml {
namespace impl {

template <typename T>
static std::vector<int64_t> SortIndexes(const T *v,
                                        int64_t num,
                                        bool descending = false) {
    std::vector<int64_t> indices(num);
    std::iota(indices.begin(), indices.end(), 0);
    if (descending) {
        std::stable_sort(indices.begin(), indices.end(),
                         [&v](int64_t i, int64_t j) { return v[i] > v[j]; });
    } else {
        std::stable_sort(indices.begin(), indices.end(),
                         [&v](int64_t i, int64_t j) { return v[i] < v[j]; });
    }
    return indices;
}

static void AllPairsIoUCPU(const float *boxes,
                           const float *scores,
                           const int64_t *sort_indices,
                           uint64_t *mask,
                           int N,
                           double nms_overlap_thresh) {
    const int num_block_cols = DIVUP(N, NMS_BLOCK_SIZE);
    const int num_block_rows = DIVUP(N, NMS_BLOCK_SIZE);

    // We need the concept of "block" since the mask is a uint64_t binary bit
    // map, and the block size is exactly 64x64. This is also consistent with
    // the CUDA implementation.
    for (int block_col_idx = 0; block_col_idx < num_block_cols;
         block_col_idx++) {
        for (int block_row_idx = 0; block_row_idx < num_block_rows;
             ++block_row_idx) {
            // Local block row size.
            const int row_size =
                    fminf(N - block_row_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);
            // Local block col size.
            const int col_size =
                    fminf(N - block_col_idx * NMS_BLOCK_SIZE, NMS_BLOCK_SIZE);

            // Comparing src and dst. In one block, the following src and dst
            // indices are compared:
            // - src: BS * block_row_idx : BS * block_row_idx + row_size
            // - dst: BS * block_col_idx : BS * block_col_idx + col_size
            //
            // With all blocks, all src and dst indices are compared.
            //
            // Result:
            // mask[i, j] is a 64-bit integer where mask[i, j][k] (k counted
            // from right) is 1 iff box[i] overlaps with box[BS*j+k].
            for (int src_idx = NMS_BLOCK_SIZE * block_row_idx;
                 src_idx < NMS_BLOCK_SIZE * block_row_idx + row_size;
                 src_idx++) {
                uint64_t t = 0;
                for (int dst_idx = NMS_BLOCK_SIZE * block_col_idx;
                     dst_idx < NMS_BLOCK_SIZE * block_col_idx + col_size;
                     dst_idx++) {
                    // Unlike the CUDA impl, both src_idx and dst_idx here are
                    // indexes to the global memory. Thus we need to compute the
                    // local index for dst_idx.
                    if (iou_bev(boxes + sort_indices[src_idx] * 5,
                                boxes + sort_indices[dst_idx] * 5) >
                        nms_overlap_thresh) {
                        t |= 1ULL << (dst_idx - NMS_BLOCK_SIZE * block_col_idx);
                    }
                }
                mask[src_idx * num_block_cols + block_col_idx] = t;
            }
        }
    }
}

// [inputs]
// boxes             : (N, 5) float32
// scores            : (N,) float32
// nms_overlap_thresh: double
//
// [return]
// keep_indices      : (M,) int64, the selected box indices
std::vector<int64_t> NmsCPUKernel(const float *boxes,
                                  const float *scores,
                                  int N,
                                  double nms_overlap_thresh) {
    std::vector<int64_t> sort_indices = SortIndexes(scores, N, true);

    const int num_block_cols = DIVUP(N, NMS_BLOCK_SIZE);

    // Call kernel. Results will be saved in masks.
    // boxes: (N, 5)
    // mask:  (N, N/BS)
    std::vector<uint64_t> mask_vec(N * num_block_cols);
    uint64_t *mask = mask_vec.data();
    AllPairsIoUCPU(boxes, scores, sort_indices.data(), mask, N,
                   nms_overlap_thresh);

    // Write to keep.
    // remv_cpu has N bits in total. If the bit is 1, the corresponding
    // box will be removed.
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
            keep_indices.push_back(sort_indices[i]);

            // Any box that overlaps with the i-th box will be removed.
            uint64_t *p = mask + i * num_block_cols;
            for (int j = block_col_idx; j < num_block_cols; j++) {
                remv_cpu[j] |= p[j];
            }
        }
    }

    return keep_indices;
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
