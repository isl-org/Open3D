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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#include "open3d/core/nns/FaissIndex.h"

#include <faiss/IndexFlat.h>

#ifdef BUILD_CUDA_MODULE
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

#include "open3d/core/Device.h"
#include "open3d/core/SizeVector.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace nns {

FaissIndex::FaissIndex() {}

FaissIndex::FaissIndex(const Tensor &dataset_points) {
    SetTensorData(dataset_points);
}

FaissIndex::~FaissIndex() {}

bool FaissIndex::SetTensorData(const Tensor &dataset_points) {
    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();

    // Check dtype.
    dataset_points_.AssertDtype(Dtype::Float32);

    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "[FaissIndex::SetTensorData] dataset_points must be "
                "2D matrix, with shape {n_dataset_points, d}.");
    }

    if (dimension == 0 || dataset_size == 0) {
        utility::LogWarning(
                "[FaissIndex::SetTensorData] Failed due to no data.");
    }

    if (dataset_points_.GetDevice().GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        res.reset(new faiss::gpu::StandardGpuResources());
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = dataset_points_.GetDevice().GetID();
        index.reset(new faiss::gpu::GpuIndexFlat(
                res.get(), dimension, faiss::MetricType::METRIC_L2, config));
#else
        utility::LogError(
                "[FaissIndex::SetTensorData] GPU Tensor is not supported when "
                "BUILD_CUDA_MODULE=OFF. Please recompile Open3D with "
                "BUILD_CUDA_MODULE=ON.");
#endif
    } else {
        index.reset(new faiss::IndexFlatL2(dimension));
    }
    float *_data_ptr = dataset_points_.GetDataPtr<float>();
    index->add(dataset_size, _data_ptr);
    return true;
}

std::pair<Tensor, Tensor> FaissIndex::SearchKnn(const Tensor &query_points,
                                                int knn) const {
    // Check dtype.
    query_points.AssertDtype(Dtype::Float32);

    // Check shape.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    if (knn <= 0) {
        utility::LogError(
                "[FaissIndex::SearchKnn] knn should be larger than 0.");
    }

    int64_t num_query_points = query_points.GetShape()[0];
    knn = std::min(knn, (int)GetDatasetSize());

    auto *data_ptr = query_points.GetDataPtr<float>();

    Tensor indices = Tensor::Empty({num_query_points * knn}, Dtype::Int64,
                                   dataset_points_.GetDevice());
    Tensor distances = Tensor::Empty({num_query_points * knn}, Dtype::Float32,
                                     dataset_points_.GetDevice());

    index->search(num_query_points, data_ptr, knn,
                  distances.GetDataPtr<float>(), indices.GetDataPtr<int64_t>());

    indices = indices.Reshape({num_query_points, knn});
    distances = distances.Reshape({num_query_points, knn});
    return std::make_pair(indices, distances);
}

std::pair<Tensor, Tensor> FaissIndex::SearchHybrid(const Tensor &query_points,
                                                   double radius,
                                                   int max_knn) const {
    // Check dtype.
    query_points.AssertDtype(Dtype::Float32);

    // Check shape.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    if (max_knn <= 0) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] max_knn should be larger than 0.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] radius should be larger than 0.");
    }

    Tensor indices;
    Tensor distances;

    std::tie(indices, distances) = SearchKnn(query_points, max_knn);

    Tensor invalid = distances.Gt(radius);
    Tensor invalid_indices = Tensor(std::vector<int64_t>({-1}), {1},
                                    Dtype::Int64, indices.GetDevice());
    Tensor invalid_distances = Tensor(std::vector<float>({-1}), {1},
                                      Dtype::Float32, distances.GetDevice());

    indices.SetItem(TensorKey::IndexTensor(invalid), invalid_indices);
    distances.SetItem(TensorKey::IndexTensor(invalid), invalid_distances);

    return std::make_pair(indices, distances);
}

}  // namespace nns
}  // namespace core
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
