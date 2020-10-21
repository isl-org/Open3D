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

int FaissIndex::GetDimension() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

size_t FaissIndex::GetDatasetSize() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

Dtype FaissIndex::GetDtype() const { return dataset_points_.GetDtype(); }

bool FaissIndex::SetTensorData(const Tensor &dataset_points) {
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "[FaissIndex::SetTensorData] dataset_points must be "
                "2D matrix, with shape {n_dataset_points, d}.");
        return false;
    }
    dataset_points_ = dataset_points.Contiguous();
    size_t dataset_size = GetDatasetSize();
    int dimension = GetDimension();
    Dtype dtype = GetDtype();

    if (dtype != Dtype::Float32) {
        utility::LogError(
                "[FaissIndex::SetTensorData] Data type must be Float32.");
        return false;
    }
    if (dimension == 0 || dataset_size == 0) {
        utility::LogWarning(
                "[FaissIndex::SetTensorData] Failed due to no data.");
        return false;
    }

    if (dataset_points_.GetBlob()->GetDevice().GetType() ==
        Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        res.reset(new faiss::gpu::StandardGpuResources());
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = dataset_points_.GetBlob()->GetDevice().GetID();
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
    float *_data_ptr =
            static_cast<float *>(dataset_points_.GetBlob()->GetDataPtr());
    index->add(dataset_size, _data_ptr);
    return true;
}

std::pair<Tensor, Tensor> FaissIndex::SearchKnn(const Tensor &query_points,
                                                int knn) const {
    if (query_points.GetDtype() != Dtype::Float32) {
        utility::LogError("[FaissIndex::SearchKnn] Data type must be Float32.");
    }
    if (query_points.NumDims() != 2) {
        utility::LogError(
                "[FaissIndex::SearchKnn] query must be 2D matrix, "
                "with shape (n_query_points, d).");
    }
    if (query_points.GetShape()[1] != GetDimension()) {
        utility::LogError(
                "[FaissIndex::SearchKnn] query has different "
                "dimension with the dataset dimension.");
    }
    if (knn <= 0) {
        utility::LogError(
                "[FaissIndex::SearchKnn] knn should be larger than 0.");
    }

    SizeVector size = query_points.GetShape();
    int64_t query_size = size[0];
    knn = std::min(knn, (int)GetDatasetSize());

    float *_data_ptr =
            static_cast<float *>(query_points.GetBlob()->GetDataPtr());

    std::vector<int64_t> indices;
    std::vector<float> distance2;
    indices.resize(knn * query_size);
    distance2.resize(knn * query_size);
    index->search(query_size, _data_ptr, knn, distance2.data(), indices.data());

    Tensor result_indices_(indices, {query_size, knn}, Dtype::Int64,
                           query_points.GetBlob()->GetDevice());
    Tensor result_distance2_(distance2, {query_size, knn}, Dtype::Float32,
                             query_points.GetBlob()->GetDevice());
    std::pair<Tensor, Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

std::pair<Tensor, Tensor> FaissIndex::SearchHybrid(const Tensor &query_points,
                                                   float radius,
                                                   int max_knn) const {
    if (query_points.GetDtype() != Dtype::Float32) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] Data type must be Float32.");
    }
    if (query_points.NumDims() != 2) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] query must be 2D matrix, "
                "with shape (n_query_points, d).");
    }
    if (query_points.GetShape()[1] != GetDimension()) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] query has different "
                "dimension with the dataset dimension.");
    }
    if (max_knn <= 0) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] max_knn should be larger than 0.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[FaissIndex::SearchHybrid] radius should be larger than 0.");
    }

    SizeVector size = query_points.GetShape();
    int64_t query_size = size[0];

    float *_data_ptr =
            static_cast<float *>(query_points.GetBlob()->GetDataPtr());

    std::vector<int64_t> indices;
    std::vector<float> distance2;
    indices.resize(max_knn * query_size);
    distance2.resize(max_knn * query_size);
    index->search(query_size, _data_ptr, max_knn, distance2.data(),
                  indices.data());

    unsigned int upper_ = max_knn * query_size;
    for (unsigned int i = 0; i < upper_; i++) {
        if (distance2[i] > radius) {
            distance2[i] = 0;
            indices[i] = -1;
        }
    }

    Tensor result_indices_(indices, {query_size, max_knn}, Dtype::Int64,
                           query_points.GetBlob()->GetDevice());
    Tensor result_distance2_(distance2, {query_size, max_knn}, Dtype::Float32,
                             query_points.GetBlob()->GetDevice());
    std::pair<Tensor, Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

}  // namespace nns
}  // namespace core
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
