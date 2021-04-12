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

#include "open3d/core/nns/NearestNeighborSearch.h"

#include "open3d/core/CoreUtil.h"
#include "open3d/utility/Console.h"

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
#ifdef WITH_FAISS
        faiss_index_.reset(new FaissIndex());
        return faiss_index_->SetTensorData(dataset_points_);
#else
        utility::LogError(
                "[NearestNeighborSearch::KnnIndex] Currently, Faiss is "
                "disabled. Please recompile Open3D with WITH_FAISS=ON.");
#endif
    } else {
        return SetIndex();
    }
};

bool NearestNeighborSearch::MultiRadiusIndex() { return SetIndex(); };

bool NearestNeighborSearch::FixedRadiusIndex(utility::optional<double> radius) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (!radius.has_value())
            utility::LogError(
                    "[NearestNeighborSearch::FixedRadiusIndex] radius is "
                    "required for GPU FixedRadiusIndex.");
#ifdef BUILD_CUDA_MODULE
        fixed_radius_index_.reset(new nns::FixedRadiusIndex());
        return fixed_radius_index_->SetTensorData(dataset_points_,
                                                  radius.value());
#else
        utility::LogError(
                "[NearestNeighborSearch::FixedRadiusIndex] FixedRadiusIndex "
                "with GPU tensor is disabled since BUILD_CUDA_MODULE is OFF. "
                "Please recompile Open3D with BUILD_CUDA_MODULE=ON.");
#endif

    } else {
        return SetIndex();
    }
}

bool NearestNeighborSearch::HybridIndex(utility::optional<double> radius) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (!radius.has_value())
            utility::LogError(
                    "[NearestNeighborSearch::HybridIndex] radius is "
                    "required for GPU HybridIndex.");
#ifdef BUILD_CUDA_MODULE
        fixed_radius_index_.reset(new nns::FixedRadiusIndex());
        return fixed_radius_index_->SetTensorData(dataset_points_,
                                                  radius.value());
#else
        utility::LogError(
                "[NearestNeighborSearch::HybridIndex] HybridIndex"
                "with GPU tensor is disabled since BUILD_CUDA_MODULE is OFF. "
                "Please recompile Open3D with BUILD_CUDA_MODULE=ON.");
#endif

    } else {
        return SetIndex();
    }
};

std::pair<Tensor, Tensor> NearestNeighborSearch::KnnSearch(
        const Tensor& query_points, int knn) {
#ifdef WITH_FAISS
    if (faiss_index_) {
        return faiss_index_->SearchKnn(query_points, knn);
    }
#endif
    if (nanoflann_index_) {
        return nanoflann_index_->SearchKnn(query_points, knn);
    } else {
        utility::LogError(
                "[NearestNeighborSearch::KnnSearch] Index is not set.");
    }
}

std::tuple<Tensor, Tensor, Tensor> NearestNeighborSearch::FixedRadiusSearch(
        const Tensor& query_points, double radius, bool sort) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (fixed_radius_index_) {
            return fixed_radius_index_->SearchRadius(query_points, radius,
                                                     sort);
        } else {
            utility::LogError(
                    "[NearsetNeighborSearch::FixedRadiusSearch] Index is not "
                    "set.");
        }
    } else {
        if (nanoflann_index_) {
            return nanoflann_index_->SearchRadius(query_points, radius);
        } else {
            utility::LogError(
                    "[NearestNeighborSearch::FixedRadiusSearch] Index is not "
                    "set.");
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> NearestNeighborSearch::MultiRadiusSearch(
        const Tensor& query_points, const Tensor& radii) {
    AssertNotCUDA(query_points);
    if (!nanoflann_index_) {
        utility::LogError(
                "[NearestNeighborSearch::MultiRadiusSearch] Index is not set.");
    }
    Dtype dtype = dataset_points_.GetDtype();
    if (dtype != query_points.GetDtype()) {
        utility::LogError(
                "[NearsetNeighborSearch::MultiRadiusSearch] reference and "
                "query have different dtype.");
    }
    if (dtype != radii.GetDtype()) {
        utility::LogError(
                "[NearsetNeighborSearch::MultiRadiusSearch] radii and data "
                "have different data type.");
    }
    return nanoflann_index_->SearchRadius(query_points, radii);
}

std::pair<Tensor, Tensor> NearestNeighborSearch::HybridSearch(
        const Tensor& query_points, double radius, int max_knn) {
    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
        if (fixed_radius_index_) {
            return fixed_radius_index_->SearchHybrid(query_points, radius,
                                                     max_knn);
        } else {
            utility::LogError(
                    "[NearestNeighborSearch::HybridSearch] Index is not set.");
        }
    } else {
        if (nanoflann_index_) {
            return nanoflann_index_->SearchHybrid(query_points, radius,
                                                  max_knn);
        } else {
            utility::LogError(
                    "[NearestNeighborSearch::HybridSearch] Index is not set.");
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
