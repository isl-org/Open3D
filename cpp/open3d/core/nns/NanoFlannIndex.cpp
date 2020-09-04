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

#include <nanoflann.hpp>

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

bool NanoFlannIndex::SetTensorData(const Tensor &dataset_points) {
    SizeVector shape = dataset_points.GetShape();
    Dtype dtype = dataset_points.GetDtype();
    if (shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SetTensorData] Tensor must be "
                "two-dimenional, with shape {n, d}.");
    }
    if (dtype != Dtype::Float64) {
        utility::LogError(
                "[NanoFlannIndex::SetTensorData] Tensor with dtype other than "
                "float64 is not supported currently.");
    }

    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();

    double *data_ptr = static_cast<double *>(dataset_points_.GetDataPtr());
    adaptor_.reset(new Adaptor<double>(dataset_size, dimension, data_ptr));

    index_.reset(new KDTree_t(GetDimension(), *adaptor_.get()));
    index_->buildIndex();
    return true;
};

std::pair<Tensor, Tensor> NanoFlannIndex::SearchKnn(const Tensor &query_points,
                                                    int knn) {
    SizeVector shape = query_points.GetShape();
    if (shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query tensor must be 2 "
                "dimensional "
                "matrix, with shape {n, d}.");
    }
    if (shape[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query tensor has different "
                "dimension "
                "with reference tensor.");
    }
    if (knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] knn should be larger than 0.");
    }
    std::vector<size_t> batch_indices;
    std::vector<double> batch_distances;
    std::vector<size_t> batch_nums;

    int64_t num_query_points = shape[0];

    for (auto i = 0; i < num_query_points; i++) {
        std::vector<size_t> single_indices(knn);
        std::vector<double> single_distances(knn);

        size_t num_results = index_->knnSearch(
                static_cast<double *>(query_points[i].GetDataPtr()),
                static_cast<size_t>(knn), single_indices.data(),
                single_distances.data());

        single_indices.resize(num_results);
        single_distances.resize(num_results);
        batch_indices.insert(batch_indices.end(), single_indices.begin(),
                             single_indices.end());
        batch_distances.insert(batch_distances.end(), single_distances.begin(),
                               single_distances.end());
        batch_nums.push_back(num_results);
    }
    if (!all_of(batch_nums.begin(), batch_nums.end(),
                [&](size_t i) { return i == batch_nums[0]; })) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] The number of neighbors are "
                "different. Something went wrong.");
    }
    if (batch_nums[0] < 1) {
        return std::make_pair(Tensor(), Tensor());
    }

    std::vector<int64_t> batch_indices2(batch_indices.begin(),
                                        batch_indices.end());
    Tensor indices(batch_indices2,
                   {num_query_points, static_cast<int64_t>(batch_nums[0])},
                   Dtype::Int64);
    Tensor distances(batch_distances,
                     {num_query_points, static_cast<int64_t>(batch_nums[0])},
                     Dtype::Float64);
    return std::make_pair(indices, distances);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, const std::vector<double> &radii) {
    SizeVector shape = query_points.GetShape();
    if (shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor must be 2 "
                "dimensional "
                "matrix, with shape {n, d}");
    }
    if (shape[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor has different "
                "dimension with reference tensor.");
    }
    std::vector<size_t> batch_indices;
    std::vector<double> batch_distances;
    std::vector<size_t> batch_nums;

    for (auto i = 0; i < shape[0]; i++) {
        double radius = radii[i];
        if (radius <= 0.0) {
            utility::LogError(
                    "[NanoFlannIndex::SearchRadius] radius should be larger "
                    "than 0.");
        }

        nanoflann::SearchParams params;
        std::vector<std::pair<size_t, double>> ret_matches;
        size_t num_matches = index_->radiusSearch(
                static_cast<double *>(query_points[i].GetDataPtr()),
                radius * radius, ret_matches, params);

        ret_matches.resize(num_matches);
        batch_nums.push_back(num_matches);
        for (auto ret = ret_matches.begin(); ret < ret_matches.end(); ret++) {
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
    Tensor indices(batch_indices2, {static_cast<int64_t>(size)}, Dtype::Int64);
    Tensor distances(batch_distances, {static_cast<int64_t>(size)},
                     Dtype::Float64);
    Tensor nums(batch_nums2, {static_cast<int64_t>(batch_nums2.size())},
                Dtype::Int64);
    return std::make_tuple(indices, distances, nums);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, double radius) {
    SizeVector shape = query_points.GetShape();
    if (shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor must be 2 "
                "dimensional "
                "matrix, with shape {n, d}.");
    }

    int64_t num_query_points = shape[0];
    std::vector<double> radii(num_query_points, radius);

    return SearchRadius(query_points, radii);
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
