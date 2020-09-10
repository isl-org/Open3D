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
        std::vector<size_t> batch_indices;
        std::vector<scalar_t> batch_distances;
        std::vector<size_t> batch_nums;

        for (auto i = 0; i < num_query_points; i++) {
            std::vector<size_t> single_indices(knn);
            std::vector<scalar_t> single_distances(knn);

            auto holder = static_cast<NanoFlannIndexHolder<L2, scalar_t> *>(
                    holder_.get());

            size_t num_results = holder->index_->knnSearch(
                    static_cast<scalar_t *>(query_points[i].GetDataPtr()),
                    static_cast<size_t>(knn), single_indices.data(),
                    single_distances.data());

            single_indices.resize(num_results);
            single_distances.resize(num_results);
            batch_indices.insert(batch_indices.end(), single_indices.begin(),
                                 single_indices.end());
            batch_distances.insert(batch_distances.end(),
                                   single_distances.begin(),
                                   single_distances.end());
            batch_nums.push_back(num_results);
        }
        if (!all_of(batch_nums.begin(), batch_nums.end(),
                    [&](size_t i) { return i == batch_nums[0]; })) {
            utility::LogError(
                    "[NanoFlannIndex::SearchKnn] The number of neighbors are "
                    "different. Something went wrong.");
        }
        if (batch_nums[0] > 0) {
            std::vector<int64_t> batch_indices2(batch_indices.begin(),
                                                batch_indices.end());
            indices = Tensor(
                    batch_indices2,
                    {num_query_points, static_cast<int64_t>(batch_nums[0])},
                    Dtype::Int64);
            distances = Tensor(
                    batch_distances,
                    {num_query_points, static_cast<int64_t>(batch_nums[0])},
                    Dtype::FromType<scalar_t>());
        }
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
                "[NanoFlannIndex::SearchRadius] query tensor and radii have "
                "different data type.");
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

    Dtype dtype = GetDtype();
    Tensor indices;
    Tensor distances;
    Tensor num_neighbors;
    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        std::vector<size_t> batch_indices;
        std::vector<scalar_t> batch_distances;
        std::vector<size_t> batch_nums;

        for (auto i = 0; i < radii.GetShape()[0]; i++) {
            scalar_t radius = radii[i].Item<scalar_t>();
            if (radius <= 0.0) {
                utility::LogError(
                        "[NanoFlannIndex::SearchRadius] radius should be "
                        "larger than 0.");
            }

            nanoflann::SearchParams params;
            std::vector<std::pair<size_t, scalar_t>> ret_matches;

            auto holder = static_cast<NanoFlannIndexHolder<L2, scalar_t> *>(
                    holder_.get());
            size_t num_matches = holder->index_->radiusSearch(
                    static_cast<scalar_t *>(query_points[i].GetDataPtr()),
                    radius * radius, ret_matches, params);

            ret_matches.resize(num_matches);
            batch_nums.push_back(num_matches);
            for (auto ret = ret_matches.begin(); ret < ret_matches.end();
                 ret++) {
                batch_indices.push_back(ret->first);
                batch_distances.push_back(ret->second);
            }
        }
        size_t size = 0;
        for (auto &s : batch_nums) {
            size += s;
        }
        std::vector<int64_t> batch_nums2(batch_nums.begin(), batch_nums.end());
        std::vector<int64_t> batch_indices2(batch_indices.begin(),
                                            batch_indices.end());
        indices = Tensor(batch_indices2, {static_cast<int64_t>(size)},
                         Dtype::Int64);
        distances = Tensor(batch_distances, {static_cast<int64_t>(size)},
                           Dtype::FromType<scalar_t>());
        num_neighbors =
                Tensor(batch_nums2, {static_cast<int64_t>(batch_nums2.size())},
                       Dtype::Int64);
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
                     {num_query_points}, Dtype::FromType<scalar_t>());
        result = SearchRadius(query_points, radii);
    });
    return result;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
