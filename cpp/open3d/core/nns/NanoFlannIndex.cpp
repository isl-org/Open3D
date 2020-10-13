// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <tbb/parallel_for.h>

#include <nanoflann.hpp>

#include "open3d/core/CoreUtil.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace nns {

NanoFlannIndex::NanoFlannIndex(){};

NanoFlannIndex::NanoFlannIndex(const Tensor &dataset_points) {
    SetTensorData(dataset_points);
};

NanoFlannIndex::~NanoFlannIndex(){};

int NanoFlannIndex::GetDimension() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

size_t NanoFlannIndex::GetDatasetSize() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

Dtype NanoFlannIndex::GetDtype() const { return dataset_points_.GetDtype(); }

bool NanoFlannIndex::SetTensorData(const Tensor &dataset_points) {
    SizeVector shape = dataset_points.GetShape();
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SetTensorData] dataset_points must be "
                "2D matrix, with shape {n_dataset_points, d}.");
        return false;
    }
    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();
    Dtype dtype = GetDtype();

    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        const scalar_t *data_ptr =
                static_cast<const scalar_t *>(dataset_points.GetDataPtr());
        holder_.reset(new NanoFlannIndexHolder<L2, scalar_t>(
                dataset_size, dimension, data_ptr));
    });
    return true;
};

std::pair<Tensor, Tensor> NanoFlannIndex::SearchKnn(const Tensor &query_points,
                                                    int knn) {
    // Check dtype.
    if (query_points.GetDtype() != GetDtype()) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] Data type mismatch {} != {}.",
                query_points.GetDtype().ToString(), GetDtype().ToString());
    }
    // Check shapes.
    if (query_points.NumDims() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query_points must be 2D matrix, "
                "with shape {n_query_points, d}.");
    }
    if (query_points.GetShape()[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query_points has different "
                "dimension with dataset_points.");
    }
    if (knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] knn should be larger than 0.");
    }

    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();

    Tensor indices;
    Tensor distances;
    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        Tensor batch_indices =
                Tensor::Full({num_query_points, knn}, -1, Dtype::Int64);
        Tensor batch_distances =
                Tensor::Full({num_query_points, knn}, -1, dtype);

        auto holder = static_cast<NanoFlannIndexHolder<L2, scalar_t> *>(
                holder_.get());

        // Parallel search.
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, num_query_points),
                [&](const tbb::blocked_range<size_t> &r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        auto single_indices = static_cast<int64_t *>(
                                batch_indices[i].GetDataPtr());
                        auto single_distances = static_cast<scalar_t *>(
                                batch_distances[i].GetDataPtr());

                        // search
                        holder->index_->knnSearch(
                                static_cast<scalar_t *>(
                                        query_points[i].GetDataPtr()),
                                static_cast<size_t>(knn), single_indices,
                                single_distances);
                    }
                });
        // Check if the number of neighbors are same.
        Tensor check_valid =
                batch_indices.Ge(0).To(Dtype::Int64).Sum({-1}, false);
        int64_t num_neighbors = check_valid[0].Item<int64_t>();
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
        const Tensor &query_points, const Tensor &radii) {
    // Check dtype.
    if (query_points.GetDtype() != GetDtype()) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] Data type mismatch {} != {}.",
                query_points.GetDtype().ToString(), GetDtype().ToString());
    }
    if (query_points.GetDtype() != radii.GetDtype()) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor and radii "
                "have different data type.");
    }
    // Check shapes.
    if (query_points.NumDims() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor must be 2 "
                "dimensional matrix, with shape {n, d}.");
    }
    if (query_points.GetShape()[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor has different "
                "dimension with reference tensor.");
    }
    if (query_points.GetShape()[0] != radii.GetShape()[0] ||
        radii.NumDims() != 1) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] radii tensor must be 1 "
                "dimensional matrix, with shape {n, }.");
    }

    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();
    Tensor indices;
    Tensor distances;
    Tensor num_neighbors;

    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        std::vector<std::vector<size_t>> batch_indices(num_query_points);
        std::vector<std::vector<scalar_t>> batch_distances(num_query_points);
        std::vector<int64_t> batch_nums;

        auto holder = static_cast<NanoFlannIndexHolder<L2, scalar_t> *>(
                holder_.get());

        nanoflann::SearchParams params;

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
                    std::vector<std::pair<int64_t, scalar_t>> ret_matches;
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        scalar_t radius = radii[i].Item<scalar_t>();

                        size_t num_results = holder->index_->radiusSearch(
                                static_cast<scalar_t *>(
                                        query_points[i].GetDataPtr()),
                                radius * radius, ret_matches, params);
                        ret_matches.resize(num_results);
                        std::vector<size_t> single_indices;
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
        std::vector<int64_t> batch_indices2;
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
        // Make result Tensors.
        int64_t total_nums = 0;
        for (auto &s : batch_nums) {
            total_nums += s;
        }
        indices = Tensor(batch_indices2, {total_nums}, Dtype::Int64);
        distances = Tensor(batch_distances2, {total_nums}, dtype);
        num_neighbors = Tensor(batch_nums, {num_query_points}, Dtype::Int64);
    });
    return std::make_tuple(indices, distances, num_neighbors);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, double radius) {
    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();
    std::tuple<Tensor, Tensor, Tensor> result;
    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        Tensor radii(std::vector<scalar_t>(num_query_points,
                                           static_cast<scalar_t>(radius)),
                     {num_query_points}, dtype);
        result = SearchRadius(query_points, radii);
    });
    return result;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
