// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/FixedRadiusIndex.h"

#include "open3d/core/Dispatch.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

FixedRadiusIndex::FixedRadiusIndex(){};

FixedRadiusIndex::FixedRadiusIndex(const Tensor &dataset_points,
                                   double radius) {
    AssertTensorDtypes(dataset_points, {Float32, Float64});
    SetTensorData(dataset_points, radius);
};

FixedRadiusIndex::FixedRadiusIndex(const Tensor &dataset_points,
                                   double radius,
                                   const Dtype &index_dtype) {
    AssertTensorDtypes(dataset_points, {Float32, Float64});
    assert(index_dtype == Int32 || index_dtype == Int64);
    SetTensorData(dataset_points, radius, index_dtype);
};

FixedRadiusIndex::~FixedRadiusIndex(){};

bool FixedRadiusIndex::SetTensorData(const Tensor &dataset_points,
                                     double radius,
                                     const Dtype &index_dtype) {
    const int64_t num_dataset_points = dataset_points.GetShape()[0];
    Tensor points_row_splits(std::vector<int64_t>({0, num_dataset_points}), {2},
                             Int64);
    return SetTensorData(dataset_points, points_row_splits, radius,
                         index_dtype);
}

bool FixedRadiusIndex::SetTensorData(const Tensor &dataset_points,
                                     const Tensor &points_row_splits,
                                     double radius,
                                     const Dtype &index_dtype) {
    AssertTensorDtypes(dataset_points, {Float32, Float64});
    assert(index_dtype == Int32 || index_dtype == Int64);
    AssertTensorDevice(points_row_splits, Device("CPU:0"));
    AssertTensorDtype(points_row_splits, Int64);

    if (radius <= 0) {
        utility::LogError("radius should be positive.");
    }
    if (dataset_points.GetShape()[0] != points_row_splits[-1].Item<int64_t>()) {
        utility::LogError(
                "dataset_points and points_row_splits have incompatible "
                "shapes.");
    }

    dataset_points_ = dataset_points.Contiguous();
    points_row_splits_ = points_row_splits.Contiguous();
    index_dtype_ = index_dtype;

    const int64_t num_dataset_points = GetDatasetSize();
    const int64_t num_batch = points_row_splits.GetShape()[0] - 1;
    const Device device = GetDevice();
    const Dtype dtype = GetDtype();

    std::vector<uint32_t> hash_table_splits(num_batch + 1, 0);
    for (int i = 0; i < num_batch; ++i) {
        int64_t num_dataset_points_i =
                points_row_splits_[i + 1].Item<int64_t>() -
                points_row_splits_[i].Item<int64_t>();
        int64_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(hash_table_size_factor * num_dataset_points_i,
                                  1),
                max_hash_tabls_size);
        hash_table_splits[i + 1] =
                hash_table_splits[i] + (uint32_t)hash_table_size;
    }

    hash_table_splits_ = Tensor(hash_table_splits, {num_batch + 1}, UInt32);
    hash_table_index_ = Tensor::Empty({num_dataset_points}, UInt32, device);
    hash_table_cell_splits_ =
            Tensor::Empty({hash_table_splits.back() + 1}, UInt32, device);

#define BUILD_PARAMETERS                                             \
    dataset_points_, radius, points_row_splits_, hash_table_splits_, \
            hash_table_index_, hash_table_cell_splits_

#define CALL_BUILD(type, fn)                \
    if (Dtype::FromType<type>() == dtype) { \
        fn<type>(BUILD_PARAMETERS);         \
        return true;                        \
    }

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        CALL_BUILD(float, BuildSpatialHashTableCUDA)
        CALL_BUILD(double, BuildSpatialHashTableCUDA)
#else
        utility::LogError(
                "-DBUILD_CUDA_MODULE=OFF. Please compile Open3d with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif
    } else {
        CALL_BUILD(float, BuildSpatialHashTableCPU)
        CALL_BUILD(double, BuildSpatialHashTableCPU)
    }
    return false;
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points, double radius, bool sort) const {
    const int64_t num_query_points = query_points.GetShape()[0];
    Tensor queries_row_splits(std::vector<int64_t>({0, num_query_points}), {2},
                              Int64);
    return SearchRadius(query_points, queries_row_splits, radius, sort);
}

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points,
        const Tensor &queries_row_splits,
        double radius,
        bool sort) const {
    const Dtype dtype = GetDtype();
    const Dtype index_dtype = GetIndexDtype();
    const Device device = GetDevice();

    // Check device and dtype.
    AssertTensorDevice(query_points, device);
    AssertTensorDtype(query_points, dtype);
    AssertTensorDevice(queries_row_splits, Device("CPU:0"));
    AssertTensorDtype(queries_row_splits, Int64);

    // Check shape.
    AssertTensorShape(query_points, {utility::nullopt, GetDimension()});
    AssertTensorShape(queries_row_splits, points_row_splits_.GetShape());

    const int64_t num_query_points = query_points.GetShape()[0];
    if (num_query_points != queries_row_splits[-1].Item<int64_t>()) {
        utility::LogError(
                "query_points and queries_row_splits have incompatible "
                "shapes.");
    }

    if (radius <= 0) {
        utility::LogError("radius should be positive.");
    }

    Tensor query_points_ = query_points.Contiguous();
    Tensor queries_row_splits_ = queries_row_splits.Contiguous();

    Tensor neighbors_index, neighbors_distance;
    Tensor neighbors_row_splits = Tensor({num_query_points + 1}, Int64, device);

#define RADIUS_PARAMETERS                                               \
    dataset_points_, query_points_, radius, points_row_splits_,         \
            queries_row_splits_, hash_table_splits_, hash_table_index_, \
            hash_table_cell_splits_, Metric::L2, false, true, sort,     \
            neighbors_index, neighbors_row_splits, neighbors_distance

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
            FixedRadiusSearchCUDA<scalar_t, int_t>(RADIUS_PARAMETERS);
        });
#else
        utility::LogError(
                "-DBUILD_CUDA_MODULE=OFF. Please compile Open3d with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif
    } else {
        DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
            FixedRadiusSearchCPU<scalar_t, int_t>(RADIUS_PARAMETERS);
        });
    }

    return std::make_tuple(neighbors_index, neighbors_distance,
                           neighbors_row_splits.To(index_dtype));
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchHybrid(
        const Tensor &query_points, double radius, int max_knn) const {
    const int64_t num_query_points = query_points.GetShape()[0];
    Tensor queries_row_splits(std::vector<int64_t>({0, num_query_points}), {2},
                              Int64);
    return SearchHybrid(query_points, queries_row_splits, radius, max_knn);
}

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchHybrid(
        const Tensor &query_points,
        const Tensor &queries_row_splits,
        double radius,
        int max_knn) const {
    const Dtype dtype = GetDtype();
    const Dtype index_dtype = GetIndexDtype();
    const Device device = GetDevice();

    // Check device and dtype.
    AssertTensorDevice(query_points, device);
    AssertTensorDtype(query_points, dtype);
    AssertTensorDevice(queries_row_splits, Device("CPU:0"));
    AssertTensorDtype(queries_row_splits, Int64);

    // Check shape.
    AssertTensorShape(query_points, {utility::nullopt, GetDimension()});
    AssertTensorShape(queries_row_splits, points_row_splits_.GetShape());

    const int64_t num_query_points = query_points.GetShape()[0];
    if (num_query_points != queries_row_splits[-1].Item<int64_t>()) {
        utility::LogError(
                "query_points and queries_row_splits have incompatible "
                "shapes.");
    }

    if (radius <= 0) {
        utility::LogError("radius should be positive.");
    }

    Tensor query_points_ = query_points.Contiguous();
    Tensor queries_row_splits_ = queries_row_splits.Contiguous();

    Tensor neighbors_index, neighbors_distance, neighbors_count;

#define HYBRID_PARAMETERS                                                \
    dataset_points_, query_points_, radius, max_knn, points_row_splits_, \
            queries_row_splits_, hash_table_splits_, hash_table_index_,  \
            hash_table_cell_splits_, Metric::L2, neighbors_index,        \
            neighbors_count, neighbors_distance

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
            HybridSearchCUDA<scalar_t, int_t>(HYBRID_PARAMETERS);
        });
#else
        utility::LogError(
                "-DBUILD_CUDA_MODULE=OFF. Please compile Open3d with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif
    } else {
        DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
            HybridSearchCPU<scalar_t, int_t>(HYBRID_PARAMETERS);
        });
    }

    return std::make_tuple(neighbors_index.View({num_query_points, max_knn}),
                           neighbors_distance.View({num_query_points, max_knn}),
                           neighbors_count.View({num_query_points}));
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
