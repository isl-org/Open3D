// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/utility/Helper.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void RadiusSearchCPU(const paddle::Tensor& points,
                     const paddle::Tensor& queries,
                     const paddle::Tensor& radii,
                     const paddle::Tensor& points_row_splits,
                     const paddle::Tensor& queries_row_splits,
                     const Metric metric,
                     const bool ignore_query_point,
                     const bool return_distances,
                     const bool normalize_distances,
                     paddle::Tensor& neighbors_index,
                     paddle::Tensor& neighbors_row_splits,
                     paddle::Tensor& neighbors_distance);

std::vector<paddle::Tensor> MultiRadiusSearch(
        paddle::Tensor& points,
        paddle::Tensor& queries,
        paddle::Tensor& radii,
        paddle::Tensor& points_row_splits,
        paddle::Tensor& queries_row_splits,
        const std::string& index_dtype,
        const std::string& metric_str,
        const bool ignore_query_point,
        const bool return_distances,
        const bool normalize_distances) {
    Metric metric = L2;
    if (metric_str == "L1") {
        metric = L1;
    } else if (metric_str == "L2") {
        metric = L2;
    } else {
        PD_CHECK(false, "metric must be one of (L1, L2) but got " + metric_str);
    }
    CHECK_TYPE(points_row_splits, paddle::DataType::INT64);
    CHECK_TYPE(queries_row_splits, paddle::DataType::INT64);
    CHECK_SAME_DTYPE(points, queries, radii);
    CHECK_SAME_DEVICE_TYPE(points, queries, radii);
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
    CHECK_SHAPE(radii, num_queries);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);

    const auto& point_type = points.dtype();

    auto place = points.place();

    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_row_splits =
            paddle::empty({queries.shape()[0] + 1},
                          paddle::DataType(ToPaddleDtype<int64_t>()), place);
    paddle::Tensor neighbors_distance;

#define FN_PARAMETERS                                                      \
    points, queries, radii, points_row_splits, queries_row_splits, metric, \
            ignore_query_point, return_distances, normalize_distances,     \
            neighbors_index, neighbors_row_splits, neighbors_distance

    if (points.is_gpu()) {
        PD_CHECK(false, "MultiRadiusSearch does not support CUDA");
    } else {
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == "int32") {
                RadiusSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                RadiusSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == "int32") {
                RadiusSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                RadiusSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return {neighbors_index, neighbors_row_splits, neighbors_distance};
    }
    // same question of fixed_radius_search
    PD_CHECK(false, "MultiRadiusSearch does not support " +
                            phi::DataTypeToString(points.dtype()) +
                            " as input for points");
    return {neighbors_index, neighbors_row_splits, neighbors_distance};
}

std::vector<paddle::DataType> MultiRadiusSearchInferDtype(
        const std::string& index_dtype) {
    paddle::DataType dtype = index_dtype == "int32" ? paddle::DataType::INT32
                                                    : paddle::DataType::INT64;
    return {dtype, paddle::DataType::INT64, dtype};
}

std::vector<std::vector<int64_t>> MultiRadiusSearchInferShape(
        std::vector<int64_t> queries_shape, const bool return_distances) {
    // this just a temp impl , all return is fake data
    // TODO(woodman3): impl real data
    int64_t neighbors_row_splits_shape = queries_shape[0] + 1;
    int64_t neighbors_distance_shape = return_distances ? 1 : 0;
    return {{neighbors_row_splits_shape},
            {neighbors_row_splits_shape},
            {neighbors_distance_shape}};
}

PD_BUILD_OP(open3d_radius_search)
        .Inputs({"points", "queries", "radii", "points_row_splits",
                 "queries_row_splits"})
        .Outputs({"neighbors_index", "neighbors_row_splits",
                  "neighbors_distance"})
        .Attrs({"index_dtype: std::string", "metric_str: std::string",
                "ignore_query_point: bool", "return_distances: bool",
                "normalize_distances: bool"})
        .SetKernelFn(PD_KERNEL(MultiRadiusSearch))
        .SetInferShapeFn(PD_INFER_SHAPE(MultiRadiusSearchInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(MultiRadiusSearchInferDtype));
