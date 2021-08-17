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

#include "open3d/core/nns/NanoFlannIndex.h"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <nanoflann.hpp>

#include "open3d/core/Dispatch.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace core {
namespace nns {

NanoFlannIndex::NanoFlannIndex(){};

NanoFlannIndex::NanoFlannIndex(const Tensor &dataset_points) {
    SetTensorData(dataset_points);
};

NanoFlannIndex::~NanoFlannIndex(){};

bool NanoFlannIndex::SetTensorData(const Tensor &dataset_points) {
    SizeVector shape = dataset_points.GetShape();
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SetTensorData] dataset_points must be "
                "2D matrix, with shape {n_dataset_points, d}.");
    }
    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();
    Dtype dtype = GetDtype();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t *data_ptr = dataset_points.GetDataPtr<scalar_t>();
        holder_.reset(new NanoFlannIndexHolder<L2, scalar_t, index_t>(
                dataset_size, dimension, data_ptr));
    });
    return true;
};

template <class T>
void NanoFlannIndex::SearchKnnSingle(const T *query_point,
                                     int knn,
                                     index_t *indices_ptr,
                                     T *distances_ptr,
                                     index_t &count) const {
    // Check dtype.
    dataset_points_.AssertDtype(Dtype::FromType<T>());
    // Check arguments.
    if (knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearcKnnSingle] knn should be larger than "
                "0.");
    }

    auto holder =
            static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder_.get());
    size_t num_neighbors = holder->index_->knnSearch(
            query_point, (size_t)knn, indices_ptr, distances_ptr);
    count = (index_t)num_neighbors;
};

std::pair<Tensor, Tensor> NanoFlannIndex::SearchKnn(const Tensor &query_points,
                                                    int knn) const {
    // Check dtype.
    query_points.AssertDtype(GetDtype());

    // Check shapes.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    if (knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] knn should be larger than 0.");
    }

    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();

    Tensor indices;
    Tensor distances;
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        Tensor batch_indices = Tensor::Full({num_query_points, knn}, -1,
                                            Dtype::FromType<index_t>());
        Tensor batch_distances =
                Tensor::Full({num_query_points, knn}, -1, dtype);

        auto holder =
                static_cast<NanoFlannIndexHolder<L2, scalar_t, index_t> *>(
                        holder_.get());

        // Parallel search.
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_query_points),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        auto single_indices =
                                batch_indices[i].GetDataPtr<index_t>();
                        auto single_distances =
                                batch_distances[i].GetDataPtr<scalar_t>();

                        // search
                        holder->index_->knnSearch(
                                query_points[i].GetDataPtr<scalar_t>(),
                                (size_t)knn, single_indices, single_distances);
                    }
                });
        // Check if the number of neighbors are same.
        Tensor check_valid = batch_indices.Ge(0)
                                     .To(Dtype::FromType<index_t>())
                                     .Sum({-1}, false);
        int64_t num_neighbors = check_valid[0].Item<index_t>();
        if (check_valid.Ne(num_neighbors).Any()) {
            utility::LogError(
                    "[NanoFlannIndex::SearchKnn] The number of neighbors are "
                    "different. Something went wrong.");
        }
        // Slice non-zero items.
        indices = batch_indices.Slice(1, 0, num_neighbors)
                          .View({num_query_points, num_neighbors});
        distances = batch_distances.Slice(1, 0, num_neighbors)
                            .View({num_query_points, num_neighbors});
    });
    return std::make_pair(indices, distances);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, const Tensor &radii, bool sort) const {
    // Check dtype.
    query_points.AssertDtype(GetDtype());
    radii.AssertDtype(GetDtype());

    // Check shapes.
    int64_t num_query_points = query_points.GetShape()[0];
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});
    radii.AssertShape({num_query_points});

    Dtype dtype = GetDtype();
    Tensor indices;
    Tensor distances;
    Tensor neighbors_row_splits;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        std::vector<std::vector<index_t>> batch_indices(num_query_points);
        std::vector<std::vector<scalar_t>> batch_distances(num_query_points);
        std::vector<int64_t> batch_nums;

        auto holder =
                static_cast<NanoFlannIndexHolder<L2, scalar_t, index_t> *>(
                        holder_.get());

        nanoflann::SearchParams params;
        params.sorted = sort;

        // Check if the raii has negative values.
        Tensor below_zero = radii.Le(0);
        if (below_zero.Any()) {
            utility::LogError(
                    "[NanoFlannIndex::SearchRadius] radius should be "
                    "larger than 0.");
        }

        // Parallel search.
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_query_points),
                [&](const tbb::blocked_range<size_t> &r) {
                    std::vector<std::pair<index_t, scalar_t>> ret_matches;
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        scalar_t radius = radii[i].Item<scalar_t>();
                        scalar_t radius_squared = radius * radius;

                        size_t num_results = holder->index_->radiusSearch(
                                query_points[i].GetDataPtr<scalar_t>(),
                                radius_squared, ret_matches, params);
                        ret_matches.resize(num_results);
                        std::vector<index_t> single_indices;
                        std::vector<scalar_t> single_distances;
                        for (auto it = ret_matches.begin();
                             it < ret_matches.end(); it++) {
                            single_indices.push_back(it->first);
                            single_distances.push_back(it->second);
                        }
                        batch_indices[i] = single_indices;
                        batch_distances[i] = single_distances;
                    }
                });

        // Flatten.
        std::vector<index_t> batch_indices2;
        std::vector<scalar_t> batch_distances2;
        for (auto i = 0; i < num_query_points; i++) {
            batch_indices2.insert(batch_indices2.end(),
                                  batch_indices[i].begin(),
                                  batch_indices[i].end());
            batch_distances2.insert(batch_distances2.end(),
                                    batch_distances[i].begin(),
                                    batch_distances[i].end());
            batch_nums.push_back(batch_indices[i].size());
        }
        std::vector<int64_t> batch_row_splits(num_query_points + 1, 0);
        utility::InclusivePrefixSum(batch_nums.data(),
                                    batch_nums.data() + batch_nums.size(),
                                    &batch_row_splits[1]);

        // Make result Tensors.
        int64_t total_nums = batch_row_splits[num_query_points];

        indices = Tensor(batch_indices2, {total_nums},
                         Dtype::FromType<index_t>());
        distances = Tensor(batch_distances2, {total_nums}, dtype);
        neighbors_row_splits =
                Tensor(batch_row_splits, {num_query_points + 1}, core::Int64);
    });
    return std::make_tuple(indices, distances, neighbors_row_splits);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, double radius, bool sort) const {
    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();
    std::tuple<Tensor, Tensor, Tensor> result;
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        Tensor radii(std::vector<scalar_t>(num_query_points, (scalar_t)radius),
                     {num_query_points}, dtype);
        result = SearchRadius(query_points, radii, sort);
    });
    return result;
};

template <class T>
void NanoFlannIndex::SearchHybridSingle(const T *query_point,
                                        double radius,
                                        int max_knn,
                                        index_t *indices_ptr,
                                        T *distances_ptr,
                                        index_t &count) const {
    // Check dtype.
    dataset_points_.AssertDtype(Dtype::FromType<T>());
    // Check arguments.
    if (max_knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearcHybridSingle] max_knn should be larger "
                "than 0.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchHybridSingle] radius should be larger "
                "than 0.");
    }

    double radius_squared = radius * radius;

    auto holder =
            static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder_.get());
    nanoflann::SearchParams params;

    std::vector<std::pair<index_t, T>> ret_matches;
    size_t num_results = holder->index_->radiusSearch(
            query_point, radius_squared, ret_matches, params);
    ret_matches.resize(num_results);

    index_t num_neighbors =
            (index_t)num_results < max_knn ? (index_t)num_results : max_knn;
    count = num_neighbors;

    int neighbour_idx = 0;
    for (auto it = ret_matches.begin();
         it < ret_matches.end() && neighbour_idx < max_knn;
         it++, neighbour_idx++) {
        indices_ptr[neighbour_idx] = it->first;
        distances_ptr[neighbour_idx] = it->second;
    }
    while (neighbour_idx < max_knn) {
        indices_ptr[neighbour_idx] = -1;
        distances_ptr[neighbour_idx] = 0;
        neighbour_idx += 1;
    }
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchHybrid(
        const Tensor &query_points, double radius, int max_knn) const {
    query_points.AssertDtype(GetDtype());
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    if (max_knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchHybrid] max_knn should be larger than "
                "0.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchHybrid] radius should be larger than "
                "0.");
    }

    double radius_squared = radius * radius;
    int64_t num_query_points = query_points.GetShape()[0];
    Tensor indices, distances, counts;
    Dtype dtype = GetDtype();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        indices = Tensor::Empty({num_query_points, max_knn},
                                Dtype::FromType<index_t>());
        auto indices_ptr = indices.GetDataPtr<index_t>();
        distances = Tensor::Empty({num_query_points, max_knn}, dtype);
        auto distances_ptr = distances.GetDataPtr<scalar_t>();
        counts = Tensor::Empty({num_query_points}, Dtype::FromType<index_t>());
        auto counts_ptr = counts.GetDataPtr<index_t>();

        auto holder =
                static_cast<NanoFlannIndexHolder<L2, scalar_t, index_t> *>(
                        holder_.get());

        nanoflann::SearchParams params;

        // Parallel search.
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_query_points),
                [&](const tbb::blocked_range<size_t> &r) {
                    std::vector<std::pair<index_t, scalar_t>> ret_matches;
                    for (size_t workload_idx = r.begin();
                         workload_idx != r.end(); ++workload_idx) {
                        int64_t result_idx = workload_idx * max_knn;

                        size_t num_results = holder->index_->radiusSearch(
                                query_points[workload_idx]
                                        .GetDataPtr<scalar_t>(),
                                radius_squared, ret_matches, params);
                        ret_matches.resize(num_results);

                        index_t result_count = (index_t)num_results;
                        result_count =
                                result_count < max_knn ? result_count : max_knn;

                        counts_ptr[workload_idx] = result_count;

                        int neighbour_idx = 0;
                        for (auto it = ret_matches.begin();
                             it < ret_matches.end() && neighbour_idx < max_knn;
                             it++, neighbour_idx++) {
                            indices_ptr[result_idx + neighbour_idx] = it->first;
                            distances_ptr[result_idx + neighbour_idx] =
                                    it->second;
                        }

                        while (neighbour_idx < max_knn) {
                            indices_ptr[result_idx + neighbour_idx] = -1;
                            distances_ptr[result_idx + neighbour_idx] = 0;
                            neighbour_idx += 1;
                        }
                    }
                });
    });
    return std::make_tuple(indices, distances, counts);
}

template void NanoFlannIndex::SearchKnnSingle(const float *query_point,
                                              int knn,
                                              index_t *indices_ptr,
                                              float *distances_ptr,
                                              index_t &count) const;

template void NanoFlannIndex::SearchKnnSingle(const double *query_point,
                                              int knn,
                                              index_t *indices_ptr,
                                              double *distances_ptr,
                                              index_t &count) const;

template void NanoFlannIndex::SearchHybridSingle(const float *query_point,
                                                 double radius,
                                                 int max_knn,
                                                 index_t *indices_ptr,
                                                 float *distances_ptr,
                                                 index_t &count) const;

template void NanoFlannIndex::SearchHybridSingle(const double *query_point,
                                                 double radius,
                                                 int max_knn,
                                                 index_t *indices_ptr,
                                                 double *distances_ptr,
                                                 index_t &count) const;
}  // namespace nns
}  // namespace core
}  // namespace open3d
