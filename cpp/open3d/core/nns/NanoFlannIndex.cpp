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

NanoFlannIndexBase::NanoFlannIndexBase(){};
NanoFlannIndexBase::~NanoFlannIndexBase(){};

template <typename T>
NanoFlannIndex<T>::NanoFlannIndex(){};

template <typename T>
NanoFlannIndex<T>::NanoFlannIndex(const core::Tensor &dataset_points) {
    SetTensorData(dataset_points);
};

template <typename T>
NanoFlannIndex<T>::~NanoFlannIndex(){};

template <typename T>
int NanoFlannIndex<T>::GetDimension() const {
    core::SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

template <typename T>
size_t NanoFlannIndex<T>::GetDatasetSize() const {
    core::SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

template <typename T>
bool NanoFlannIndex<T>::SetTensorData(const core::Tensor &dataset_points) {
    core::SizeVector shape = dataset_points.GetShape();
    if (shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SetTensorData] dataset_points must be "
                "2D matrix, with shape {n, d}.");
        return false;
    }
    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();

    const T *data_ptr = static_cast<const T *>(dataset_points.GetDataPtr());
    adaptor_.reset(new Adaptor<T>(dataset_size, dimension, data_ptr));

    index_.reset(new KDTree_t(dimension, *adaptor_.get()));
    index_->buildIndex();
    return true;
};

template <typename T>
std::pair<core::Tensor, core::Tensor> NanoFlannIndex<T>::SearchKnn(
        const core::Tensor &query_points, int knn) {
    core::SizeVector query_shape = query_points.GetShape();
    if (query_shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query_points must be 2D matrix, "
                "with shape {n, d}.");
    }
    if (query_shape[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] query_points has different "
                "dimension "
                "with dataset_points.");
    }
    if (knn <= 0) {
        utility::LogError(
                "[NanoFlannIndex::SearchKnn] knn should be larger than 0.");
    }
    std::vector<size_t> batch_indices;
    std::vector<T> batch_distances;
    std::vector<size_t> batch_nums;

    int64_t num_query_points = query_shape[0];

    for (auto i = 0; i < num_query_points; i++) {
        std::vector<size_t> single_indices(knn);
        std::vector<T> single_distances(knn);

        size_t num_results = index_->knnSearch(
                static_cast<T *>(query_points[i].GetDataPtr()),
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
        return std::make_pair(core::Tensor(), core::Tensor());
    }

    std::vector<int64_t> batch_indices2(batch_indices.begin(),
                                        batch_indices.end());
    core::Tensor indices(
            batch_indices2,
            {num_query_points, static_cast<int64_t>(batch_nums[0])},
            core::Dtype::Int64);
    core::Tensor distances(
            batch_distances,
            {num_query_points, static_cast<int64_t>(batch_nums[0])},
            core::Dtype::FromType<T>());
    return std::make_pair(indices, distances);
};

template <typename T>
std::tuple<core::Tensor, core::Tensor, core::Tensor>
NanoFlannIndex<T>::SearchRadius(const core::Tensor &query_points,
                                const core::Tensor &radii) {
    core::SizeVector query_shape = query_points.GetShape();
    core::SizeVector radii_shape = radii.GetShape();
    // check if query_points is 2D matrix
    if (query_shape.size() != 2) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor must be 2 "
                "dimensional "
                "matrix, with shape {n, d}.");
    }
    // check if query_points has same dimension with dataset_points
    if (query_shape[1] != GetDimension()) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor has different "
                "dimension with reference tensor.");
    }
    // check if radii is 1D matrix and have same number of elements with
    // query_points
    if (query_shape[0] != radii_shape[0] || radii_shape.size() != 1) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] radii tensor must be 1 "
                "dimensional matrix, with shape {n, }.");
    }
    // check if query_points and radii have same data type
    if (query_points.GetDtype() != radii.GetDtype()) {
        utility::LogError(
                "[NanoFlannIndex::SearchRadius] query tensor and radii have "
                "different data type.");
    }
    std::vector<size_t> batch_indices;
    std::vector<T> batch_distances;
    std::vector<size_t> batch_nums;

    for (auto i = 0; i < radii_shape[0]; i++) {
        T radius = *static_cast<T *>(radii[i].GetDataPtr());
        if (radius <= 0.0) {
            utility::LogError(
                    "[NanoFlannIndex::SearchRadius] radius should be larger "
                    "than 0.");
        }

        nanoflann::SearchParams params;
        std::vector<std::pair<size_t, T>> ret_matches;
        size_t num_matches = index_->radiusSearch(
                static_cast<T *>(query_points[i].GetDataPtr()), radius * radius,
                ret_matches, params);

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
    core::Tensor indices(batch_indices2, {static_cast<int64_t>(size)},
                         core::Dtype::Int64);
    core::Tensor distances(batch_distances, {static_cast<int64_t>(size)},
                           core::Dtype::FromType<T>());
    core::Tensor nums(batch_nums2, {static_cast<int64_t>(batch_nums2.size())},
                      core::Dtype::Int64);
    return std::make_tuple(indices, distances, nums);
};

template <typename T>
std::tuple<core::Tensor, core::Tensor, core::Tensor>
NanoFlannIndex<T>::SearchRadius(const core::Tensor &query_points, T radius) {
    core::SizeVector query_shape = query_points.GetShape();

    int64_t num_query_points = query_shape[0];
    core::Tensor radii(std::vector<T>(num_query_points, radius),
                       {num_query_points}, core::Dtype::FromType<T>());
    // std::vector<T> radii(num_query_points, radius);

    return SearchRadius(query_points, radii);
};

template class NanoFlannIndex<double>;
template class NanoFlannIndex<float>;
}  // namespace nns
}  // namespace core
}  // namespace open3d
