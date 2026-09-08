// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <algorithm>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/TorchOpen3DBridge.h"
#include "open3d/utility/Helper.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
#include <c10/cuda/CUDAStream.h>

#include "open3d/core/nns/KnnIndex.h"
#endif

namespace o3dnns = open3d::core::nns;

template <class T, class TIndex>
void KnnSearchCPU(const torch::Tensor& points,
                  const torch::Tensor& queries,
                  const int64_t k,
                  const torch::Tensor& points_row_splits,
                  const torch::Tensor& queries_row_splits,
                  const o3dnns::Metric metric,
                  const bool ignore_query_point,
                  const bool return_distances,
                  torch::Tensor& neighbors_index,
                  torch::Tensor& neighbors_row_splits,
                  torch::Tensor& neighbors_distance);

#ifdef BUILD_SYCL_MODULE
// Implemented in KnnSearchOpKernelSYCL.cpp (compiled with the SYCL
// compiler). L1 metric and ignore_query_point are not supported by the SYCL
// KNN kernel; callers must check this before dispatching here.
template <class T, class TIndex>
void KnnSearchSYCL(const torch::Tensor& points,
                   const torch::Tensor& queries,
                   const int64_t k,
                   const torch::Tensor& points_row_splits,
                   const torch::Tensor& queries_row_splits,
                   torch::Tensor& neighbors_index,
                   torch::Tensor& neighbors_row_splits,
                   torch::Tensor& neighbors_distance);
#endif

void ConvertToGlobalIndices(torch::Tensor& neighbors_index,
                            const torch::Tensor& points_row_splits,
                            const torch::Tensor& queries_row_splits,
                            int64_t k) {
    int64_t neighbors_offset = 0;
    const int64_t batch_size = points_row_splits.size(0) - 1;
    for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t points_begin = points_row_splits[i].item<int64_t>();
        const int64_t num_points =
                points_row_splits[i + 1].item<int64_t>() - points_begin;
        const int64_t num_queries = queries_row_splits[i + 1].item<int64_t>() -
                                    queries_row_splits[i].item<int64_t>();
        const int64_t num_neighbors = std::min(k, num_points) * num_queries;
        if (num_neighbors > 0) {
            neighbors_index
                    .slice(0, neighbors_offset,
                           neighbors_offset + num_neighbors)
                    .add_(points_begin);
        }
        neighbors_offset += num_neighbors;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> KnnSearch(
        torch::Tensor points,
        torch::Tensor queries,
        const int64_t k,
        torch::Tensor points_row_splits,
        torch::Tensor queries_row_splits,
        torch::ScalarType index_dtype,
        const std::string& metric_str,
        const bool ignore_query_point,
        const bool return_distances) {
    o3dnns::Metric metric = o3dnns::L2;
    if (metric_str == "L1") {
        metric = o3dnns::L1;
    } else if (metric_str == "L2") {
        metric = o3dnns::L2;
    } else {
        TORCH_CHECK(false,
                    "metric must be one of (L1, L2) but got " + metric_str);
    }
    TORCH_CHECK(k > 0, "k must be greater than zero");
    CHECK_TYPE(points_row_splits, kInt64);
    CHECK_TYPE(queries_row_splits, kInt64);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);
    TORCH_CHECK(index_dtype == torch::kInt32 || index_dtype == torch::kInt64,
                "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.to(torch::kCPU);
    queries_row_splits = queries_row_splits.to(torch::kCPU);
    points = points.contiguous();
    queries = queries.contiguous();
    points_row_splits = points_row_splits.contiguous();
    queries_row_splits = queries_row_splits.contiguous();

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);

    const auto& point_type = points.dtype();

    auto device = points.device().type();
    auto device_idx = points.device().index();

    torch::Tensor neighbors_index;
    torch::Tensor neighbors_row_splits = torch::empty(
            {queries.size(0) + 1},
            torch::dtype(ToTorchDtype<int64_t>()).device(device, device_idx));
    torch::Tensor neighbors_distance;

#define FN_PARAMETERS                                                  \
    points, queries, k, points_row_splits, queries_row_splits, metric, \
            ignore_query_point, return_distances, neighbors_index,     \
            neighbors_row_splits, neighbors_distance

#define GPU_FN_PARAMETERS                                      \
    points, queries, k, points_row_splits, queries_row_splits, \
            neighbors_index, neighbors_row_splits, neighbors_distance

    if (points.is_cuda() || points.is_xpu()) {
        // The CUDA/SYCL KNN kernels only support the L2 metric and do not
        // support ignore_query_point.
        TORCH_CHECK(metric == o3dnns::L2,
                    "KnnSearch on CUDA/XPU only supports the L2 metric.");
        TORCH_CHECK(
                !ignore_query_point,
                "KnnSearch on CUDA/XPU does not support ignore_query_point.");
    }

    if (points.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        using open3d::ml::torch_bridge::Open3DToTorchTensor;
        using open3d::ml::torch_bridge::TorchToOpen3DTensor;

        open3d::core::Tensor points_ = TorchToOpen3DTensor(points);
        open3d::core::Tensor queries_ = TorchToOpen3DTensor(queries);
        open3d::core::Tensor points_row_splits_ =
                TorchToOpen3DTensor(points_row_splits);
        open3d::core::Tensor queries_row_splits_ =
                TorchToOpen3DTensor(queries_row_splits);
        open3d::core::Tensor neighbors_index_, neighbors_distance_;
        // KnnSearchCUDA requires neighbors_row_splits to be pre-allocated by
        // the caller (it only fills it in), matching the pattern used by
        // KnnIndex::SearchKnn.
        open3d::core::Tensor neighbors_row_splits_ =
                open3d::core::Tensor::Empty({queries.size(0) + 1},
                                            open3d::core::Int64,
                                            points_.GetDevice());

        cudaStream_t user_stream =
                c10::cuda::getCurrentCUDAStream(device_idx).stream();

#define CUDA_FN_PARAMETERS                                                \
    points_, points_row_splits_, queries_, queries_row_splits_, int(k),   \
            neighbors_index_, neighbors_row_splits_, neighbors_distance_, \
            user_stream

        if (CompareTorchDtype<float>(point_type)) {
            if (index_dtype == torch::kInt32) {
                o3dnns::KnnSearchCUDA<float, int32_t>(CUDA_FN_PARAMETERS);
            } else {
                o3dnns::KnnSearchCUDA<float, int64_t>(CUDA_FN_PARAMETERS);
            }
        } else {
            if (index_dtype == torch::kInt32) {
                o3dnns::KnnSearchCUDA<double, int32_t>(CUDA_FN_PARAMETERS);
            } else {
                o3dnns::KnnSearchCUDA<double, int64_t>(CUDA_FN_PARAMETERS);
            }
        }
#undef CUDA_FN_PARAMETERS

        torch::Tensor neighbors_index =
                Open3DToTorchTensor(neighbors_index_)
                        .reshape({neighbors_index_.NumElements()});
        ConvertToGlobalIndices(neighbors_index, points_row_splits,
                               queries_row_splits, k);

        // KnnSearchCUDA's batch_size==1 fast path returns a 2D
        // [num_queries, k] view; flatten to the 1D ragged layout that this
        // op's contract (and the CPU path) uses.
        return std::make_tuple(
                neighbors_index, Open3DToTorchTensor(neighbors_row_splits_),
                Open3DToTorchTensor(neighbors_distance_)
                        .reshape({neighbors_distance_.NumElements()}));
#else
        TORCH_CHECK(false, "KnnSearch was not compiled with CUDA support.")
#endif
    } else if (points.is_xpu()) {
#ifdef BUILD_SYCL_MODULE
        if (CompareTorchDtype<float>(point_type)) {
            if (index_dtype == torch::kInt32) {
                KnnSearchSYCL<float, int32_t>(GPU_FN_PARAMETERS);
            } else {
                KnnSearchSYCL<float, int64_t>(GPU_FN_PARAMETERS);
            }
        } else {
            if (index_dtype == torch::kInt32) {
                KnnSearchSYCL<double, int32_t>(GPU_FN_PARAMETERS);
            } else {
                KnnSearchSYCL<double, int64_t>(GPU_FN_PARAMETERS);
            }
        }
        neighbors_index = neighbors_index.reshape({neighbors_index.numel()});
        ConvertToGlobalIndices(neighbors_index, points_row_splits,
                               queries_row_splits, k);

        // KnnSearchSYCL's batch_size==1 fast path returns a 2D
        // [num_queries, k] view; flatten to the 1D ragged layout that this
        // op's contract (and the CPU path) uses.
        return std::make_tuple(
                neighbors_index, neighbors_row_splits,
                neighbors_distance.reshape({neighbors_distance.numel()}));
#else
        TORCH_CHECK(false, "KnnSearch was not compiled with SYCL support.")
#endif
    } else if (points.is_cpu()) {
        if (CompareTorchDtype<float>(point_type)) {
            if (index_dtype == torch::kInt32) {
                KnnSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == torch::kInt32) {
                KnnSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return std::make_tuple(neighbors_index, neighbors_row_splits,
                               neighbors_distance);
    }
    TORCH_CHECK(false, "KnnSearch does not support " + points.toString() +
                               " as input for points")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

const char* knn_fn_format =
        "open3d::knn_search(Tensor points, Tensor queries, int "
        "k, Tensor points_row_splits, Tensor queries_row_splits, ScalarType "
        "index_dtype=%d,"
        "str metric=\"L2\", bool ignore_query_point=False, bool "
        "return_distances=False) -> "
        "(Tensor neighbors_index, Tensor "
        "neighbors_row_splits, Tensor neighbors_distance)";

static auto registry = torch::RegisterOperators(
        open3d::utility::FormatString(knn_fn_format, int(c10::ScalarType::Int)),
        &KnnSearch);
