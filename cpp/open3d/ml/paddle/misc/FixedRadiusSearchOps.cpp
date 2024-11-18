// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/Dtype.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/utility/Helper.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchCPU(const paddle::Tensor& points,
                          const paddle::Tensor& queries,
                          double radius,
                          const paddle::Tensor& points_row_splits,
                          const paddle::Tensor& queries_row_splits,
                          const paddle::Tensor& hash_table_splits,
                          const paddle::Tensor& hash_table_index,
                          const paddle::Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          paddle::Tensor& neighbors_index,
                          paddle::Tensor& neighbors_row_splits,
                          paddle::Tensor& neighbors_distance);
#ifdef BUILD_CUDA_MODULE
template <class T, class TIndex>
void FixedRadiusSearchCUDA(const paddle::Tensor& points,
                           const paddle::Tensor& queries,
                           double radius,
                           const paddle::Tensor& points_row_splits,
                           const paddle::Tensor& queries_row_splits,
                           const paddle::Tensor& hash_table_splits,
                           const paddle::Tensor& hash_table_index,
                           const paddle::Tensor& hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           paddle::Tensor& neighbors_index,
                           paddle::Tensor& neighbors_row_splits,
                           paddle::Tensor& neighbors_distance);
#endif

std::vector<paddle::Tensor> FixedRadiusSearch(
        paddle::Tensor& points,
        paddle::Tensor& queries,
        paddle::Tensor& points_row_splits,
        paddle::Tensor& queries_row_splits,
        paddle::Tensor& hash_table_splits,
        paddle::Tensor& hash_table_index,
        paddle::Tensor& hash_table_cell_splits,
        double radius,
        const std::string& index_dtype,
        const std::string& metric_str,
        const bool ignore_query_point,
        const bool return_distances) {
    Metric metric = L2;
    if (metric_str == "L1") {
        metric = L1;
    } else if (metric_str == "L2") {
        metric = L2;
    } else if (metric_str == "Linf") {
        metric = Linf;
    } else {
        PD_CHECK(false,
                 "metric must be one of (L1, L2, Linf) but got " + metric_str);
    }
    CHECK_TYPE(points_row_splits, paddle::DataType::INT64);
    CHECK_TYPE(queries_row_splits, paddle::DataType::INT64);
    CHECK_TYPE(hash_table_splits, paddle::DataType::INT32);
    CHECK_TYPE(hash_table_index, paddle::DataType::INT32);
    CHECK_TYPE(hash_table_cell_splits, paddle::DataType::INT32);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);
    // PD_CHECK(index_dtype == paddle::DataType::INT32 || index_dtype ==
    // paddle::DataType::INT64,
    PD_CHECK(index_dtype == "int32" || index_dtype == "int64",
             "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.copy_to(paddle::CPUPlace(), false);
    queries_row_splits = queries_row_splits.copy_to(paddle::CPUPlace(), false);
    hash_table_splits = hash_table_splits.copy_to(paddle::CPUPlace(), false);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(hash_table_index, num_points);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_cell_splits, num_cells + 1);

    const auto& point_type = points.dtype();

    auto place = points.place();

    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_row_splits =
            paddle::empty({queries.shape()[0] + 1},
                          paddle::DataType(ToPaddleDtype<int64_t>()), place);
    paddle::Tensor neighbors_distance;

#define FN_PARAMETERS                                                      \
    points, queries, radius, points_row_splits, queries_row_splits,        \
            hash_table_splits, hash_table_index, hash_table_cell_splits,   \
            metric, ignore_query_point, return_distances, neighbors_index, \
            neighbors_row_splits, neighbors_distance

    if (points.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == "int32") {
                FixedRadiusSearchCUDA<float, int32_t>(FN_PARAMETERS);
            } else {
                FixedRadiusSearchCUDA<float, int64_t>(FN_PARAMETERS);
            }
            return {neighbors_index, neighbors_row_splits, neighbors_distance};
        }
#else
        PD_CHECK(false, "FixedRadiusSearch was not compiled with CUDA support");
#endif
    } else {
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == "int32") {
                FixedRadiusSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                FixedRadiusSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == "int32") {
                FixedRadiusSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                FixedRadiusSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return {neighbors_index, neighbors_row_splits, neighbors_distance};
    }

    // in torch the name is ToString, but paddle not have this function
    PD_CHECK(false, "FixedRadiusSearch does not support " +
                            phi::DataTypeToString(points.dtype()) +
                            " as input for points");
    return std::vector<paddle::Tensor>();
}

std::vector<paddle::DataType> FixedRadiusSearchInferDtype(
        const std::string& index_dtype) {
    paddle::DataType dtype = index_dtype == "int32" ? paddle::DataType::INT32
                                                    : paddle::DataType::INT64;
    return {dtype, paddle::DataType::INT64, dtype};
}

std::vector<std::vector<int64_t>> FixedRadiusSearchInferShape(
        std::vector<int64_t> queries_shape, const bool return_distances) {
    // this just a temp impl , all return is fake data
    // TODO(woodman3): impl real data
    int64_t neighbors_row_splits_shape = queries_shape[0] + 1;
    int64_t neighbors_distance_shape = return_distances ? 1 : 0;
    return {{neighbors_row_splits_shape},
            {neighbors_row_splits_shape},
            {neighbors_distance_shape}};
}

PD_BUILD_OP(open3d_fixed_radius_search)
        .Inputs({"points", "queries", "points_row_splits", "queries_row_splits",
                 "hash_table_splits", "hash_table_index",
                 "hash_table_cell_splits"})
        .Outputs({"neighbors_index", "neighbors_row_splits",
                  "neighbors_distance"})
        .Attrs({
                "radius: double",
                "index_dtype:std::string",
                "metric_str: std::string",
                "ignore_query_point: bool",
                "return_distances: bool",
        })
        .SetKernelFn(PD_KERNEL(FixedRadiusSearch))
        .SetInferShapeFn(PD_INFER_SHAPE(FixedRadiusSearchInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(FixedRadiusSearchInferDtype));
