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

#include "open3d/core/nns/NearestNeighborSearch.h"

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

NearestNeighborSearch::~NearestNeighborSearch(){};

bool NearestNeighborSearch::SetIndex() {
    nanoflann_index_.reset(new NanoFlannIndex());
    return nanoflann_index_->SetTensorData(dataset_points_);
};

bool NearestNeighborSearch::KnnIndex() {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        knn_index_.reset(new nns::KnnIndex());
        return knn_index_->SetTensorData(dataset_points_);
#else
        utility::LogError(
                "-DBUILD_CUDA_MODULE=OFF. Please recompile Open3D with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif
    } else {
        return SetIndex();
    }
};

bool NearestNeighborSearch::MultiRadiusIndex() { return SetIndex(); };

bool NearestNeighborSearch::FixedRadiusIndex(utility::optional<double> radius) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (!radius.has_value())
            utility::LogError("radius is required for GPU FixedRadiusIndex.");
#ifdef BUILD_CUDA_MODULE
        fixed_radius_index_.reset(new nns::FixedRadiusIndex());
        return fixed_radius_index_->SetTensorData(dataset_points_,
                                                  radius.value());
#else
        utility::LogError(
                "FixedRadiusIndex with GPU tensor is disabled since "
                "-DBUILD_CUDA_MODULE=OFF. Please recompile Open3D with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif

    } else {
        return SetIndex();
    }
}

bool NearestNeighborSearch::HybridIndex(utility::optional<double> radius) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (!radius.has_value())
            utility::LogError("radius is required for GPU HybridIndex.");
#ifdef BUILD_CUDA_MODULE
        fixed_radius_index_.reset(new nns::FixedRadiusIndex());
        return fixed_radius_index_->SetTensorData(dataset_points_,
                                                  radius.value());
#else
        utility::LogError(
                "-DBUILD_CUDA_MODULE=OFF. Please recompile Open3D with "
                "-DBUILD_CUDA_MODULE=ON.");
#endif

    } else {
        return SetIndex();
    }
};

std::pair<Tensor, Tensor> NearestNeighborSearch::KnnSearch(
        const Tensor& query_points, int knn) {
    AssertTensorDevice(query_points, dataset_points_.GetDevice());

    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (knn_index_) {
            return knn_index_->SearchKnn(query_points, knn);
        } else {
            utility::LogError("Index is not set.");
        }
    } else {
        if (nanoflann_index_) {
            return nanoflann_index_->SearchKnn(query_points, knn);
        } else {
            utility::LogError("Index is not set.");
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> NearestNeighborSearch::FixedRadiusSearch(
        const Tensor& query_points, double radius, bool sort) {
    AssertTensorDevice(query_points, dataset_points_.GetDevice());

    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (fixed_radius_index_) {
            return fixed_radius_index_->SearchRadius(query_points, radius,
                                                     sort);
        } else {
            utility::LogError("Index is not set.");
        }
    } else {
        if (nanoflann_index_) {
            return nanoflann_index_->SearchRadius(query_points, radius);
        } else {
            utility::LogError("Index is not set.");
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> NearestNeighborSearch::MultiRadiusSearch(
        const Tensor& query_points, const Tensor& radii) {
    AssertNotCUDA(query_points);
    AssertTensorDtype(query_points, dataset_points_.GetDtype());
    AssertTensorDtype(radii, dataset_points_.GetDtype());

    if (!nanoflann_index_) {
        utility::LogError("Index is not set.");
    }
    return nanoflann_index_->SearchRadius(query_points, radii);
}

std::tuple<Tensor, Tensor, Tensor> NearestNeighborSearch::HybridSearch(
        const Tensor& query_points, double radius, int max_knn) {
    AssertTensorDevice(query_points, dataset_points_.GetDevice());

    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (fixed_radius_index_) {
            return fixed_radius_index_->SearchHybrid(query_points, radius,
                                                     max_knn);
        } else {
            utility::LogError("Index is not set.");
        }
    } else {
        if (nanoflann_index_) {
            return nanoflann_index_->SearchHybrid(query_points, radius,
                                                  max_knn);
        } else {
            utility::LogError("Index is not set.");
        }
    }
}

void NearestNeighborSearch::AssertNotCUDA(const Tensor& t) const {
    if (t.GetDevice().GetType() == Device::DeviceType::CUDA) {
        utility::LogError(
                "TODO: NearestNeighborSearch does not support CUDA tensor "
                "yet.");
    }
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
