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
#include "open3d/core/nns/KnnFaiss.h"

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

KnnFaiss::KnnFaiss() {}

KnnFaiss::KnnFaiss(const Tensor &tensor) { SetTensorData(tensor); }

KnnFaiss::~KnnFaiss() {}

bool KnnFaiss::SetTensorData(const Tensor &tensor) {
    SizeVector size = tensor.GetShape();
    dimension_ = size[1];
    dataset_size_ = size[0];

    if (tensor.GetDtype() != Dtype::Float32) {
        utility::LogError(
                "[KnnFaiss::SetTensorData] Data type must be Float32.");
        return false;
    }
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KnnFaiss::SetTensorData] Failed due to no data.");
        return false;
    }

    if (tensor.GetBlob()->GetDevice().GetType() == Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        res.reset(new faiss::gpu::StandardGpuResources());
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = tensor.GetBlob()->GetDevice().GetID();
        index.reset(new faiss::gpu::GpuIndexFlat(
                res.get(), dimension_, faiss::MetricType::METRIC_L2, config));
#else
        utility::LogError(
                "[KnnFaiss::SetTensorData] GPU Tensor is not supported when "
                "BUILD_CUDA_MODULE=OFF. Please recompile Open3D with "
                "BUILD_CUDA_MODULE=ON.")
#endif
    } else {
        index.reset(new faiss::IndexFlatL2(dimension_));
    }
    float *_data_ptr = static_cast<float *>(tensor.GetBlob()->GetDataPtr());
    index->add(dataset_size_, _data_ptr);
    return true;
}

std::pair<Tensor, Tensor> KnnFaiss::SearchKnn(const Tensor &query,
                                              int knn) const {
    SizeVector size = query.GetShape();
    size_t query_dim = size[1];
    size_t query_size = size[0];
    knn = std::min(knn, (int)dataset_size_);

    if (query.GetDtype() != Dtype::Float32) {
        utility::LogError("[KnnFaiss::SearchKnn] Data type must be Float32.");
    }
    if (query.NumDims() != 2) {
        utility::LogError(
                "[KnnFaiss::SearchKnn] query must be 2D matrix, "
                "with shape (n_query_points, d).");
    }
    if (query_dim != dimension_) {
        utility::LogError(
                "[KnnFaiss::SearchKnn] query has different "
                "dimension with the dataset dimension.");
    }
    if (knn <= 0) {
        utility::LogError("[KnnFaiss::SearchKnn] knn should be larger than 0.");
    }

    float *_data_ptr = static_cast<float *>(query.GetBlob()->GetDataPtr());

    std::vector<long> indices;
    std::vector<float> distance2;
    indices.resize(knn * query_size);
    distance2.resize(knn * query_size);
    index->search(query_size, _data_ptr, knn, distance2.data(), indices.data());

    Tensor result_indices_(indices, {(long int)query_size, knn}, Dtype::Int64,
                           query.GetBlob()->GetDevice());
    Tensor result_distance2_(distance2, {(long int)query_size, knn},
                             Dtype::Float32, query.GetBlob()->GetDevice());
    std::pair<Tensor, Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

std::pair<Tensor, Tensor> KnnFaiss::SearchHybrid(const Tensor &query,
                                                 float radius,
                                                 int max_knn) const {
    SizeVector size = query.GetShape();
    size_t query_dim = size[1];
    size_t query_size = size[0];

    if (query.GetDtype() != Dtype::Float32) {
        utility::LogError(
                "[KnnFaiss::SearchHybrid] Data type must be Float32.");
    }
    if (query.NumDims() != 2) {
        utility::LogError(
                "[KnnFaiss::SearchHybrid] query must be 2D matrix, "
                "with shape (n_query_points, d).");
    }
    if (query_dim != dimension_) {
        utility::LogError(
                "[KnnFaiss::SearchHybrid] query has different "
                "dimension with dataset_points.");
    }
    if (max_knn <= 0) {
        utility::LogError(
                "[KnnFaiss::SearchHybrid] max_knn should be larger than 0.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[KnnFaiss::SearchHybrid] radius should be larger than 0.");
    }

    float *_data_ptr = static_cast<float *>(query.GetBlob()->GetDataPtr());

    std::vector<long> indices;
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

    Tensor result_indices_(indices, {(long int)query_size, max_knn},
                           Dtype::Int64, query.GetBlob()->GetDevice());
    Tensor result_distance2_(distance2, {(long int)query_size, max_knn},
                             Dtype::Float32, query.GetBlob()->GetDevice());
    std::pair<Tensor, Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

}  // namespace nns
}  // namespace core
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
