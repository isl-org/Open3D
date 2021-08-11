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

#include "open3d/core/nns/KnnIndex.h"

#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

KnnIndex::KnnIndex(){};

KnnIndex::KnnIndex(const Tensor& dataset_points) {
    SetTensorData(dataset_points);
}

KnnIndex::~KnnIndex(){};

bool KnnIndex::SetTensorData(const Tensor& dataset_points) {
    int64_t num_dataset_points = dataset_points.GetShape()[0];
    Tensor points_row_splits(std::vector<int64_t>({0, num_dataset_points}), {2},
                             Int64);
    return SetTensorData(dataset_points, points_row_splits);
}

bool KnnIndex::SetTensorData(const Tensor& dataset_points,
                             const Tensor& points_row_splits) {
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "KnnIndex::SetTensorData dataset_points must be 2D matri,x "
                "with shape {n_dataset_points,d}.");
    }
    if (dataset_points.GetShape()[0] <= 0 ||
        dataset_points.GetShape()[1] <= 0) {
        utility::LogError("KnnIndex::SetTensorData Failed due to no data.");
    }
    if (dataset_points.GetShape()[0] != points_row_splits[-1].Item<int64_t>()) {
        utility::LogError(
                "KnnIndex::SetTensorData dataset_points and points_row_splits "
                "have incompatible shapes.");
    }

    if (dataset_points.GetDevice().GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        dataset_points_ = dataset_points.Contiguous();
        points_row_splits_ = points_row_splits.Contiguous();
        return true;
#else
        utility::LogError(
                "KnnIndex::SetTensorData GPU Tensor is not supported when "
                "BUILD_CUDA_MODULE=OFF. Please recompile Open3d With "
                "BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        utility::LogError(
                "KnnIndex::SetTensorData CPU Tensor i not supported in "
                "KnnIndex. Please use NanoFlannIndex instread.");
    }
    return false;
}

std::pair<Tensor, Tensor> KnnIndex::SearchKnn(const Tensor& query_points,
                                              int knn) const {
    int64_t num_query_points = query_points.GetShape()[0];
    Tensor queries_row_splits(std::vector<int64_t>({0, num_query_points}), {2},
                              Int64);
    return SearchKnn(query_points, queries_row_splits, knn);
}

std::pair<Tensor, Tensor> KnnIndex::SearchKnn(const Tensor& query_points,
                                              const Tensor& queries_row_splits,
                                              int knn) const {
    Dtype dtype = GetDtype();
    Device device = GetDevice();

    query_points.AssertDtype(dtype);
    query_points.AssertDevice(device);
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});
    if (query_points.GetShape()[0] != queries_row_splits[-1].Item<int64_t>()) {
        utility::LogError(
                "KnnIndex::SearchKnn query_points and queries_row_splits have "
                "incompatible shapes.");
    }
    if (knn <= 0) {
        utility::LogError("KnnIndex::SearchKnn knn should be larger than 0.");
    }

    Tensor query_points_ = query_points.Contiguous();
    Tensor queries_row_splits_ = queries_row_splits.Contiguous();

    Tensor neighbors_index, neighbors_distance;

#define FN_PARAMETERS                                                        \
    dataset_points_, points_row_splits_, query_points_, queries_row_splits_, \
            knn, neighbors_index, neighbors_distance

#define CALL(type, fn)                                              \
    if (Dtype::FromType<type>() == dtype) {                         \
        fn<type>(FN_PARAMETERS);                                    \
        return std::make_pair(neighbors_index, neighbors_distance); \
    }

    if (device.GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        CALL(float, KnnSearchCUDA)
        CALL(double, KnnSearchCUDA)
#else
        utility::LogError(
                "KnnIndex::SearchKnn BUILD_CUDA_MODULE is OFF. "
                "Please compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        utility::LogError(
                "KnnIndex::SearchKnn BUILD_CUDA_MODULE is OFF. "
                "Please compile Open3d with BUILD_CUDA_MODULE=ON.");
    }
    return std::make_pair(neighbors_index, neighbors_distance);
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
