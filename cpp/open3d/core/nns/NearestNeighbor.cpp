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

#include "open3d/core/nns/NearestNeighbor.h"

#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace nns {

NearestNeighbor::~NearestNeighbor(){};

template <class T>
NanoFlannIndex<T>* NearestNeighbor::cast_index() {
    return static_cast<NanoFlannIndex<T>*>(index_.get());
}

bool NearestNeighbor::SetIndex() {
    switch (data_.GetDtype()) {
        case core::Dtype::Float64:
            index_.reset(new NanoFlannIndex<double>());
            return cast_index<double>()->SetTensorData(data_);
        case core::Dtype::Float32:
            index_.reset(new NanoFlannIndex<float>());
            return cast_index<float>()->SetTensorData(data_);
        default:
            utility::LogError("undefined data type");
    }
};
bool NearestNeighbor::KnnIndex() { return SetIndex(); };
bool NearestNeighbor::RadiusIndex() { return SetIndex(); };
bool NearestNeighbor::FixedRadiusIndex() { return SetIndex(); }
bool NearestNeighbor::HybridIndex() { return SetIndex(); };

std::pair<core::Tensor, core::Tensor> NearestNeighbor::KnnSearch(
        const core::Tensor& query, int knn) {
    if (!index_) {
        utility::LogError("[NearestNeighbor::KnnSearch] Index is not set");
    }
    if (data_.GetDtype() != query.GetDtype()) {
        utility::LogError(
                "[NearestNeighbor::KnnSearch] reference and query have "
                "different dtype");
    }
    switch (data_.GetDtype()) {
        case core::Dtype::Float64:
            return cast_index<double>()->SearchKnn(query, knn);
        case core::Dtype::Float32:
            return cast_index<float>()->SearchKnn(query, knn);
        default:
            utility::LogError("undefined data type");
    }
}

template <typename T>
std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::RadiusSearch(const core::Tensor& query, T* radii) {
    if (!index_) {
        utility::LogError("[NearestNeighbor::RadiusSearch] Index is not set");
    }
    if (data_.GetDtype() != query.GetDtype()) {
        utility::LogError(
                "[NearestNeighbor::RadiusSearch] reference and query have "
                "different dtype");
    }
    if (data_.GetDtype() != core::DtypeUtil::FromType<T>()) {
        utility::LogError(
                "[NearestNeighbor::RadiusSearch] radii and data have different "
                "data type");
    }
    return cast_index<T>()->SearchRadius(query, radii);
}

template <typename T>
std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::FixedRadiusSearch(const core::Tensor& query, T radius) {
    if (!index_) {
        utility::LogError(
                "[NearestNeighbor::FixedRadiusSearch] Index is not set");
    }
    if (data_.GetDtype() != query.GetDtype()) {
        utility::LogError(
                "[NearestNeighbor::FixedRadiusSearch] reference and query have "
                "different dtype");
    }
    if (data_.GetDtype() != core::DtypeUtil::FromType<T>()) {
        utility::LogError(
                "[NearestNeighbor::FixedRadiusSearch] radius and data have "
                "different data type");
    }
    return cast_index<T>()->SearchRadius(query, radius);
}

template <typename T>
std::pair<core::Tensor, core::Tensor> NearestNeighbor::HybridSearch(
        const core::Tensor& query, T radius, int max_knn) {
    if (!index_) {
        utility::LogError("[NearestNeighbor::HybridSearch] Index is not set");
    }
    if (data_.GetDtype() != query.GetDtype()) {
        utility::LogError(
                "[NearestNeighbor::HybridSearch] reference and query have "
                "different dtype");
    }
    if (data_.GetDtype() != core::DtypeUtil::FromType<T>()) {
        utility::LogError(
                "[NearestNeighbor::HybridSearch] radius and data have "
                "different data type");
    }

    std::pair<core::Tensor, core::Tensor> result =
            cast_index<T>()->SearchKnn(query, max_knn);

    core::Tensor indices = result.first;
    core::Tensor distances = result.second;
    core::SizeVector size = distances.GetShape();

    std::vector<int64_t> indices_vec = indices.ToFlatVector<int64_t>();
    std::vector<T> distances_vec = distances.ToFlatVector<T>();

    for (unsigned int i = 0; i < distances_vec.size(); i++) {
        if (distances_vec[i] > radius) {
            distances_vec[i] = 0;
            indices_vec[i] = -1;
        }
    }

    core::Tensor indices_new(indices_vec, size, core::Dtype::Int64);
    core::Tensor distances_new(distances_vec, size,
                               core::DtypeUtil::FromType<T>());
    return std::make_pair(indices_new, distances_new);
}

template std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::RadiusSearch(const core::Tensor& query, double* radii);

template std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::RadiusSearch(const core::Tensor& query, float* radii);

template std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::FixedRadiusSearch(const core::Tensor& query, double radius);

template std::tuple<core::Tensor, core::Tensor, core::Tensor>
NearestNeighbor::FixedRadiusSearch(const core::Tensor& query, float radius);

template std::pair<core::Tensor, core::Tensor> NearestNeighbor::HybridSearch(
        const core::Tensor& query, double radius, int max_knn);

template std::pair<core::Tensor, core::Tensor> NearestNeighbor::HybridSearch(
        const core::Tensor& query, float radius, int max_knn);
}  // namespace nns
}  // namespace core
}  // namespace open3d
