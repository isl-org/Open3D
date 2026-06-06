// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

namespace {

template <typename scalar_t>
// Launches contiguous index_add over dim0:
//   dst[index[i], ...] += src[i, ...]
// Contract:
// - `index_ptr`, `src_ptr`, and `dst_ptr` point to contiguous buffers.
// - `index_length` is the length of `index_ptr` and the leading dimension of
//   `src_ptr`.
// - `broadcasting_elems` is the flattened product of non-reduction dimensions.
// - `dst_ptr` has enough rows to address all index values.
void LaunchIndexAddContiguousSYCLKernel(sycl::queue& queue,
                                        const int64_t* index_ptr,
                                        const scalar_t* src_ptr,
                                        scalar_t* dst_ptr,
                                        int64_t index_length,
                                        int64_t broadcasting_elems) {
    if (index_length <= 0 || broadcasting_elems <= 0) {
        return;
    }

    auto ceil_div = [](int64_t a, int64_t b) -> int64_t {
        return (a + b - 1) / b;
    };
    auto round_up = [](int64_t x, int64_t m) -> int64_t {
        return ((x + m - 1) / m) * m;
    };

    // 2D launch configuration:
    // - X dimension tiles columns (broadcasting_elems).
    // - Y dimension tiles reduction rows (index_length).
    //
    // Each work-group processes TILE_ROWS rows and WG_X columns. Within a row
    // tile, consecutive runs of identical destination indices are reduced into
    // one atomic add per (column, run), reducing atomic pressure while
    // preserving index_add semantics.
    constexpr int WG_X = 256;
    constexpr int TILE_ROWS = 8;
    const int64_t num_row_tiles = ceil_div(index_length, int64_t(TILE_ROWS));
    const int64_t global_x = round_up(broadcasting_elems, int64_t(WG_X));
    sycl::nd_range<2> launch(sycl::range<2>(num_row_tiles, global_x),
                             sycl::range<2>(1, WG_X));

    queue.submit([&](sycl::handler& cgh) {
             sycl::local_accessor<int64_t, 1> l_idx(sycl::range<1>(TILE_ROWS),
                                                    cgh);

             cgh.parallel_for(
                     launch,
                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(
                             16)]] {
                         const int lid_x = int(it.get_local_id(1));
                         const int64_t group_y = it.get_group(0);
                         const int64_t col = it.get_global_id(1);
                         if (col >= broadcasting_elems) {
                             return;
                         }

                         const int64_t row_base = group_y * int64_t(TILE_ROWS);

                         if (lid_x < TILE_ROWS) {
                             const int64_t r = row_base + lid_x;
                             l_idx[lid_x] = (r < index_length) ? index_ptr[r]
                                                               : int64_t(-1);
                         }
                         it.barrier(sycl::access::fence_space::local_space);

                         int run_start = 0;
                         while (run_start < TILE_ROWS) {
                             const int64_t dst_row = l_idx[run_start];
                             if (dst_row < 0) {
                                 break;
                             }

                             int run_end = run_start + 1;
                             while (run_end < TILE_ROWS &&
                                    l_idx[run_end] == dst_row) {
                                 ++run_end;
                             }

                             scalar_t sum = scalar_t(0);
                             for (int rr = run_start; rr < run_end; ++rr) {
                                 const int64_t src_row = row_base + int64_t(rr);
                                 if (src_row < index_length) {
                                     const int64_t workload_idx =
                                             src_row * broadcasting_elems + col;
                                     sum += src_ptr[workload_idx];
                                 }
                             }

                             const int64_t dst_idx =
                                     dst_row * broadcasting_elems + col;
                             sycl::atomic_ref<scalar_t,
                                              sycl::memory_order::relaxed,
                                              sycl::memory_scope::device>
                                     aref(dst_ptr[dst_idx]);
                             aref += sum;

                             run_start = run_end;
                         }
                     });
         }).wait_and_throw();
}

}  // namespace

void IndexAddSYCL_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst) {
    // index: [N,], src: [N, D], dst: [M, D]
    // This kernel assumes contiguous layout for fast linear indexing.
    // Non-contiguous tensors are materialized as contiguous before launch.
    const Tensor index_contiguous = index.Contiguous();
    const Tensor src_contiguous = src.Contiguous();
    Tensor dst_contiguous = dst.Contiguous();

    // Index is simply a 1D contiguous tensor.
    auto index_ptr = index_contiguous.GetDataPtr<int64_t>();

    int64_t broadcasting_elems = 1;
    for (int64_t d = 1; d < src_contiguous.NumDims(); ++d) {
        broadcasting_elems *= src_contiguous.GetShape(d);
    }

    const int64_t index_length = index_contiguous.GetLength();

    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(src.GetDevice());

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        LaunchIndexAddContiguousSYCLKernel<scalar_t>(
                queue, index_ptr, src_contiguous.GetDataPtr<scalar_t>(),
                dst_contiguous.GetDataPtr<scalar_t>(), index_length,
                broadcasting_elems);
    });

    // If dst is non-contiguous, write back from the contiguous temporary.
    if (!dst.IsContiguous()) {
        dst.CopyFrom(dst_contiguous);
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
