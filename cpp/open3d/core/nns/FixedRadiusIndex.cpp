// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/nns/FixedRadiusIndex.h"

#include "open3d/core/Dispatch.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

FixedRadiusIndex::FixedRadiusIndex(){};

FixedRadiusIndex::FixedRadiusIndex(const Tensor &dataset_points,
                                   double radius) {
    SetTensorData(dataset_points, radius);
};

FixedRadiusIndex::~FixedRadiusIndex(){};

bool FixedRadiusIndex::SetTensorData(const Tensor &dataset_points,
                                     double radius) {
    if (radius <= 0) {
        utility::LogError(
                "[FixedRadiusIndex::SetTensorData] radius should be positive.");
    }
    dataset_points_ = dataset_points.Contiguous();
    Device device = GetDevice();
    Dtype dtype = GetDtype();

    int64_t num_dataset_points = GetDatasetSize();
    int64_t hash_table_size = std::min<int64_t>(
            std::max<int64_t>(hash_table_size_factor * num_dataset_points, 1),
            max_hash_tabls_size);

    std::vector<int64_t> points_row_splits({0, num_dataset_points});
    points_row_splits_ = Tensor(points_row_splits, {2}, Int64);
    std::vector<uint32_t> hash_table_splits({0, (uint32_t)hash_table_size});
    hash_table_splits_ = Tensor(hash_table_splits, {2}, UInt32);

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

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        CALL_BUILD(float, BuildSpatialHashTableCUDA)
        CALL_BUILD(double, BuildSpatialHashTableCUDA)
#else
        utility::LogError(
                "FixedRadiusIndex::SetTensorData BUILD_CUDA_MODULE is OFF. "
                "Please compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        CALL_BUILD(float, BuildSpatialHashTableCPU)
        CALL_BUILD(double, BuildSpatialHashTableCPU)
    }
    return false;
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points, double radius, bool sort) const {
    Dtype dtype = GetDtype();
    Device device = GetDevice();

    // Check dtype.
    query_points.AssertDtype(dtype);

    // Check shape.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    // Check device.
    query_points.AssertDevice(device);

    if (radius <= 0) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] radius should be positive.");
    }

    Tensor query_points_ = query_points.Contiguous();
    int64_t num_query_points = query_points_.GetShape()[0];

    std::vector<int64_t> queries_row_splits({0, num_query_points});
    Tensor queries_row_splits_(queries_row_splits, {2}, Int64);

    Tensor neighbors_index, neighbors_distance;
    Tensor neighbors_row_splits =
            Tensor({num_query_points + 1}, core::Int64, device);

#define RADIUS_PARAMETERS                                               \
    dataset_points_, query_points_, radius, points_row_splits_,         \
            queries_row_splits_, hash_table_splits_, hash_table_index_, \
            hash_table_cell_splits_, Metric::L2, false, true, sort,     \
            neighbors_index, neighbors_row_splits, neighbors_distance

#define CALL_RADIUS(type, fn)               \
    if (Dtype::FromType<type>() == dtype) { \
        fn<type>(RADIUS_PARAMETERS);        \
    }

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        CALL_RADIUS(float, FixedRadiusSearchCUDA)
        CALL_RADIUS(double, FixedRadiusSearchCUDA)
#else
        utility::LogError(
                "FixedRadiusIndex::SearchRadius BUILD_CUDA_MODULE is OFF. "
                "Please compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        CALL_RADIUS(float, FixedRadiusSearchCPU)
        CALL_RADIUS(double, FixedRadiusSearchCPU)
    }

    return std::make_tuple(neighbors_index, neighbors_distance,
                           neighbors_row_splits);
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchHybrid(
        const Tensor &query_points, double radius, int max_knn) const {
    Dtype dtype = GetDtype();
    Device device = GetDevice();

    // Check dtype.
    query_points.AssertDtype(dtype);

    // Check shape.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    // Check device.
    query_points.AssertDevice(device);

    if (radius <= 0) {
        utility::LogError(
                "[FixedRadiusIndex::SearchHybrid] radius should be positive.");
    }

    Tensor query_points_ = query_points.Contiguous();
    int64_t num_query_points = query_points_.GetShape()[0];

    std::vector<int64_t> queries_row_splits({0, num_query_points});
    Tensor queries_row_splits_(queries_row_splits, {2}, Int64);

    Tensor neighbors_index, neighbors_distance, neighbors_count;

#define HYBRID_PARAMETERS                                                \
    dataset_points_, query_points_, radius, max_knn, points_row_splits_, \
            queries_row_splits_, hash_table_splits_, hash_table_index_,  \
            hash_table_cell_splits_, Metric::L2, neighbors_index,        \
            neighbors_count, neighbors_distance

#define CALL_HYBRID(type, fn)               \
    if (Dtype::FromType<type>() == dtype) { \
        fn<type>(HYBRID_PARAMETERS);        \
    }

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        CALL_HYBRID(float, HybridSearchCUDA)
        CALL_HYBRID(double, HybridSearchCUDA)
#else
        utility::LogError(
                "FixedRadiusIndex::SearchHybrid BUILD_CUDA_MODULE is OFF. "
                "Please compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        CALL_HYBRID(float, HybridSearchCPU)
        CALL_HYBRID(double, HybridSearchCPU)
    }

    return std::make_tuple(neighbors_index.View({num_query_points, max_knn}),
                           neighbors_distance.View({num_query_points, max_knn}),
                           neighbors_count.View({num_query_points}));
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
