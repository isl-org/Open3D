// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/utility/Helper.h"
#include "paddle/extension.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void KnnSearchCPU(const paddle::Tensor& points,
                  const paddle::Tensor& queries,
                  const int64_t k,
                  const paddle::Tensor& points_row_splits,
                  const paddle::Tensor& queries_row_splits,
                  const Metric metric,
                  const bool ignore_query_point,
                  const bool return_distances,
                  paddle::Tensor& neighbors_index,
                  paddle::Tensor& neighbors_row_splits,
                  paddle::Tensor& neighbors_distance);

std::vector<paddle::Tensor> KnnSearch(paddle::Tensor& points,
                                      paddle::Tensor& queries,
                                      paddle::Tensor& points_row_splits,
                                      paddle::Tensor& queries_row_splits,
                                      const int64_t k,
                                      const std::string& index_dtype,
                                      const std::string& metric_str,
                                      const bool ignore_query_point,
                                      const bool return_distances) {
    Metric metric = L2;
    if (metric_str == "L1") {
        metric = L1;
    } else if (metric_str == "L2") {
        metric = L2;
    } else {
        PD_CHECK(false, "metric must be one of (L1, L2) but got " + metric_str);
    }
    PD_CHECK(k > 0, "k must be greater than zero");
    CHECK_TYPE(points_row_splits, phi::DataType::INT64);
    CHECK_TYPE(queries_row_splits, phi::DataType::INT64);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);
    PD_CHECK(index_dtype == "int32" || index_dtype == "int64",
             "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.copy_to(paddle::CPUPlace(), false);
    queries_row_splits = queries_row_splits.copy_to(paddle::CPUPlace(), false);

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

    auto place = points.place();

    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_row_splits =
            paddle::empty({queries.shape()[0] + 1},
                          paddle::DataType(ToPaddleDtype<int64_t>()), place);
    paddle::Tensor neighbors_distance;

#define FN_PARAMETERS                                                  \
    points, queries, k, points_row_splits, queries_row_splits, metric, \
            ignore_query_point, return_distances, neighbors_index,     \
            neighbors_row_splits, neighbors_distance

    if (points.is_gpu()) {
        PD_CHECK(false, "KnnSearch does not support CUDA");
    } else {
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == "int32") {
                KnnSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == "int32") {
                KnnSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return {neighbors_index, neighbors_row_splits, neighbors_distance};
    }
    PD_CHECK(false, "KnnSearch does not support " +
                            phi::DataTypeToString(points.dtype()) +
                            " as input for points");
    return std::vector<paddle::Tensor>();
}

std::vector<paddle::DataType> KnnSearchInferDtype(
        const std::string& index_dtype) {
    paddle::DataType dtype = index_dtype == "int32" ? paddle::DataType::INT32
                                                    : paddle::DataType::INT64;
    return {dtype, paddle::DataType::INT64, dtype};
}

std::vector<std::vector<int64_t>> KnnSearchInferShape(
        std::vector<int64_t> queries_shape, const bool return_distances) {
    int64_t neighbors_row_splits_shape = queries_shape[0] + 1;
    int64_t neighbors_distance_shape = return_distances ? 1 : 0;
    return {{neighbors_row_splits_shape},
            {neighbors_row_splits_shape},
            {neighbors_distance_shape}};
}

PD_BUILD_OP(open3d_knn_search)
        .Inputs({"points", "queries", "points_row_splits",
                 "queries_row_splits"})
        .Outputs({"neighbors_index", "neighbors_row_splits",
                  "neighbors_distance"})
        .Attrs({
                "k: int64_t",
                "index_dtype:std::string",
                "metric_str: std::string",
                "ignore_query_point: bool",
                "return_distances: bool",
        })
        .SetKernelFn(PD_KERNEL(KnnSearch))
        .SetInferShapeFn(PD_INFER_SHAPE(KnnSearchInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(KnnSearchInferDtype));
