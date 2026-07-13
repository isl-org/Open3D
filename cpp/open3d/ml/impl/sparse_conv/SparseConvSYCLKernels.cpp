// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/impl/sparse_conv/SparseConvSYCLKernels.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {
// Work-group size: one work-group processes one output point; work-items
// stride over input channels / neighbors. Matches the CUDA kernel's
// BLOCKSIZE=32 (one warp per output point).
constexpr size_t kWGSize = 32;
}  // namespace

template <class TReal, class TIndex, class TKernelIndex>
void FillColumnSYCL(sycl::queue& queue,
                    TReal* columns,
                    int in_channels,
                    TIndex begin_idx,
                    TIndex end_idx,
                    TIndex num_out,
                    TIndex num_inp,
                    const TReal* const inp_features,
                    const TReal* const inp_importance,
                    size_t neighbors_index_size,
                    const TIndex* const neighbors_index,
                    const TKernelIndex* const neighbors_kernel_index,
                    const TReal* const neighbors_importance,
                    const int64_t* const neighbors_row_splits,
                    const int num_kernel_elements,
                    bool normalize) {
    const TIndex num_columns = end_idx - begin_idx;
    if (num_columns <= 0) return;

    queue.fill(columns, TReal(0),
               size_t(num_kernel_elements) * in_channels * num_columns)
            .wait();

    const bool point_importance = inp_importance != nullptr;
    const bool neighbor_importance = neighbors_importance != nullptr;

    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(num_columns * kWGSize),
                                       sycl::range<1>(kWGSize)),
                     [=](sycl::nd_item<1> item) {
                         const TIndex out_idx =
                                 begin_idx +
                                 static_cast<TIndex>(item.get_group(0));
                         if (out_idx >= end_idx) return;

                         const TIndex col_idx = out_idx - begin_idx;
                         TReal* out_column =
                                 columns + size_t(num_kernel_elements) *
                                                   in_channels * col_idx;
                         const int64_t neighbor_start =
                                 neighbors_row_splits[out_idx];
                         const int64_t neighbor_end =
                                 neighbors_row_splits[out_idx + 1];

                         auto group = item.get_group();
                         const size_t lid = item.get_local_id(0);
                         const size_t lsize = item.get_local_range(0);

                         TReal normalizer = TReal(0);
                         if (normalize) {
                             if (neighbor_importance) {
                                 TReal local_sum = TReal(0);
                                 for (int64_t n_idx = neighbor_start +
                                                      static_cast<int64_t>(lid);
                                      n_idx < neighbor_end;
                                      n_idx += static_cast<int64_t>(lsize)) {
                                     local_sum += neighbors_importance[n_idx];
                                 }
                                 normalizer = sycl::reduce_over_group(
                                         group, local_sum, sycl::plus<TReal>());
                             } else {
                                 normalizer =
                                         TReal(neighbor_end - neighbor_start);
                             }
                         }

                         for (int64_t n_idx = neighbor_start;
                              n_idx < neighbor_end; ++n_idx) {
                             const TIndex inp_idx = neighbors_index[n_idx];
                             const TReal n_importance =
                                     neighbor_importance
                                             ? neighbors_importance[n_idx]
                                             : TReal(1);
                             const int kernel_idx =
                                     neighbors_kernel_index[n_idx];

                             TReal importance = TReal(1);
                             if (point_importance)
                                 importance = inp_importance[inp_idx];
                             if (neighbor_importance)
                                 importance *= n_importance;
                             if (normalize && normalizer != 0)
                                 importance /= normalizer;

                             for (int ic = static_cast<int>(lid);
                                  ic < in_channels;
                                  ic += static_cast<int>(lsize)) {
                                 out_column[kernel_idx * in_channels + ic] =
                                         importance *
                                         inp_features[size_t(inp_idx) *
                                                              in_channels +
                                                      ic];
                             }
                         }
                     });
         }).wait_and_throw();
}

template <class TReal, class TIndex, class TKernelIndex>
void FillColumnTransposeSYCL(sycl::queue& queue,
                             TReal* columns,
                             int in_channels,
                             TIndex begin_idx,
                             TIndex end_idx,
                             TIndex num_out,
                             TIndex num_inp,
                             const TReal* const inp_features,
                             const TReal* const inp_neighbors_importance_sum,
                             const int64_t* const inp_neighbors_prefix_sum,
                             size_t neighbors_index_size,
                             const TIndex* const neighbors_index,
                             const TKernelIndex* const neighbors_kernel_index,
                             const TReal* const neighbors_importance,
                             const int64_t* const neighbors_row_splits,
                             const int num_kernel_elements,
                             bool normalize) {
    const TIndex num_columns = end_idx - begin_idx;
    if (num_columns <= 0) return;

    queue.fill(columns, TReal(0),
               size_t(num_kernel_elements) * in_channels * num_columns)
            .wait();

    const bool neighbor_importance = neighbors_importance != nullptr;

    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(num_columns * kWGSize),
                                       sycl::range<1>(kWGSize)),
                     [=](sycl::nd_item<1> item) {
                         const TIndex out_idx =
                                 begin_idx +
                                 static_cast<TIndex>(item.get_group(0));
                         if (out_idx >= end_idx) return;

                         const TIndex col_idx = out_idx - begin_idx;
                         TReal* out_column =
                                 columns + size_t(num_kernel_elements) *
                                                   in_channels * col_idx;
                         const int64_t neighbor_start =
                                 neighbors_row_splits[out_idx];
                         const int64_t neighbor_end =
                                 neighbors_row_splits[out_idx + 1];

                         const size_t lid = item.get_local_id(0);
                         const size_t lsize = item.get_local_range(0);

                         for (int64_t n_idx = neighbor_start;
                              n_idx < neighbor_end; ++n_idx) {
                             const TIndex inp_idx = neighbors_index[n_idx];
                             const int kernel_idx =
                                     neighbors_kernel_index[n_idx];

                             TReal normalizer = TReal(1);
                             if (normalize) {
                                 if (inp_neighbors_importance_sum) {
                                     if (inp_neighbors_importance_sum
                                                 [inp_idx] != 0)
                                         normalizer /=
                                                 inp_neighbors_importance_sum
                                                         [inp_idx];
                                 } else {
                                     const int64_t inp_neighbor_start =
                                             inp_neighbors_prefix_sum[inp_idx];
                                     const int64_t inp_neighbor_end =
                                             inp_idx + 1 < num_inp
                                                     ? inp_neighbors_prefix_sum
                                                               [inp_idx + 1]
                                                     : static_cast<int64_t>(
                                                               neighbors_index_size);
                                     const int64_t num_inp_neighbors =
                                             inp_neighbor_end -
                                             inp_neighbor_start;
                                     if (num_inp_neighbors > 0)
                                         normalizer /= TReal(num_inp_neighbors);
                                 }
                             }

                             for (int ic = static_cast<int>(lid);
                                  ic < in_channels;
                                  ic += static_cast<int>(lsize)) {
                                 TReal infeat =
                                         inp_features[size_t(inp_idx) *
                                                              in_channels +
                                                      ic];
                                 if (neighbor_importance)
                                     infeat *= neighbors_importance[n_idx];
                                 if (normalize) infeat *= normalizer;

                                 sycl::atomic_ref<TReal,
                                                  sycl::memory_order::relaxed,
                                                  sycl::memory_scope::device,
                                                  sycl::access::address_space::
                                                          global_space>
                                         out_ref(out_column
                                                         [kernel_idx *
                                                                  in_channels +
                                                          ic]);
                                 out_ref.fetch_add(infeat);
                             }
                         }
                     });
         }).wait_and_throw();
}

#define INSTANTIATE(TREAL, TINDEX, TKERNELINDEX)                              \
    template void FillColumnSYCL<TREAL, TINDEX, TKERNELINDEX>(                \
            sycl::queue & queue, TREAL * columns, int in_channels,            \
            TINDEX begin_idx, TINDEX end_idx, TINDEX num_out, TINDEX num_inp, \
            const TREAL* const inp_features,                                  \
            const TREAL* const inp_importance, size_t neighbors_index_size,   \
            const TINDEX* const neighbors_index,                              \
            const TKERNELINDEX* const neighbors_kernel_index,                 \
            const TREAL* const neighbors_importance,                          \
            const int64_t* const neighbors_row_splits,                        \
            const int num_kernel_elements, bool normalize);                   \
    template void FillColumnTransposeSYCL<TREAL, TINDEX, TKERNELINDEX>(       \
            sycl::queue & queue, TREAL * columns, int in_channels,            \
            TINDEX begin_idx, TINDEX end_idx, TINDEX num_out, TINDEX num_inp, \
            const TREAL* const inp_features,                                  \
            const TREAL* const inp_neighbors_importance_sum,                  \
            const int64_t* const inp_neighbors_prefix_sum,                    \
            size_t neighbors_index_size, const TINDEX* const neighbors_index, \
            const TKERNELINDEX* const neighbors_kernel_index,                 \
            const TREAL* const neighbors_importance,                          \
            const int64_t* const neighbors_row_splits,                        \
            const int num_kernel_elements, bool normalize);

INSTANTIATE(float, int32_t, int16_t)
INSTANTIATE(float, int32_t, uint8_t)

#undef INSTANTIATE

}  // namespace impl
}  // namespace ml
}  // namespace open3d
