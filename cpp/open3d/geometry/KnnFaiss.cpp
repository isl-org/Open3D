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
#include "open3d/geometry/KnnFaiss.h"

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_factory.h>

#include "open3d/core/Device.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/HalfEdgeTriangleMesh.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace geometry {

KnnFaiss::KnnFaiss() {}

KnnFaiss::KnnFaiss(const Eigen::MatrixXd &data) { SetMatrixData(data); }

KnnFaiss::KnnFaiss(const Geometry &geometry) { SetGeometry(geometry); }

KnnFaiss::KnnFaiss(const pipelines::registration::Feature &feature) {
    SetFeature(feature);
}

KnnFaiss::KnnFaiss(const core::Tensor &tensor) { SetTensorData(tensor); }

KnnFaiss::~KnnFaiss() {}

bool KnnFaiss::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KnnFaiss::SetTensorData(const core::Tensor &tensor) {
    core::SizeVector size = tensor.GetShape();
    dimension_ = size[1];
    dataset_size_ = size[0];

    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KnnFaiss::SetTensorData] Failed due to no data.");
        return false;
    }

    data_.resize(dataset_size_ * dimension_);
    if (tensor.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA) {
        res.reset(new faiss::gpu::StandardGpuResources());
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = tensor.GetBlob()->GetDevice().GetID();
        index.reset(new faiss::gpu::GpuIndexFlat(
                res.get(), dimension_, faiss::MetricType::METRIC_L2, config));
    } else {
        index.reset(new faiss::IndexFlatL2(dimension_));
    }
    float *_data_ptr = static_cast<float *>(tensor.GetBlob()->GetDataPtr());
    index->add(dataset_size_, _data_ptr);
    return true;
}

bool KnnFaiss::SetTensorData2(const core::Tensor &tensor, const char *description_in, bool support_on_gpu, long train_size){
    core::SizeVector size = tensor.GetShape();
    dimension_ = size[1];
    dataset_size_ = size[0];
    support_on_gpu_  = support_on_gpu;

    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KnnFaiss::SetTensorData2] Failed due to no data.");
        return false;
    }

    data_.resize(dataset_size_ * dimension_);
    faiss::Index *tmp_index = faiss::index_factory(dimension_, description_in); // cpu_index
    if (tensor.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA) {
        if (support_on_gpu){
            res.reset(new faiss::gpu::StandardGpuResources());
            int _device_id = tensor.GetBlob()->GetDevice().GetID();
            index.reset(faiss::gpu::index_cpu_to_gpu(res.get(), _device_id, tmp_index));
        }
        else{
            core::Tensor tensor = tensor.Copy(core::Device("CPU:0"));
            index.reset(tmp_index);
        }
    } 
    else {
        index.reset(tmp_index);
    }
    if(!index->is_trained){
        core::Tensor xt = tensor.Slice(dimension_, 0, train_size, 1);
        float *_train_ptr = static_cast<float *>(tensor.GetBlob()->GetDataPtr());
        index->train(train_size, _train_ptr);
    }
    float *_data_ptr = static_cast<float *>(tensor.GetBlob()->GetDataPtr());
    index->add(dataset_size_, _data_ptr);
    return true;
}

bool KnnFaiss::SetGeometry(const Geometry &geometry) {
    switch (geometry.GetGeometryType()) {
        case Geometry::GeometryType::PointCloud:
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)((const PointCloud &)geometry)
                            .points_.data(),
                    3, ((const PointCloud &)geometry).points_.size()));
        case Geometry::GeometryType::TriangleMesh:
        case Geometry::GeometryType::HalfEdgeTriangleMesh:
            return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                    (const double *)((const TriangleMesh &)geometry)
                            .vertices_.data(),
                    3, ((const TriangleMesh &)geometry).vertices_.size()));
        case Geometry::GeometryType::Image:
        case Geometry::GeometryType::Unspecified:
        default:
            utility::LogWarning(
                    "[KnnFaiss::SetGeometry] Unsupported Geometry type.");
            return false;
    }
}

bool KnnFaiss::SetFeature(const pipelines::registration::Feature &feature) {
    return SetMatrixData(feature.data_);
}

template <typename T>
int KnnFaiss::Search(const T &query,
                     const KDTreeSearchParam &param,
                     std::vector<long> &indices,
                     std::vector<float> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KnnFaiss::SearchKNN(const T &query,
                        int knn,
                        std::vector<long> &indices,
                        std::vector<float> &distance2) const {
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }
    std::vector<float> tmp_query(query.size());
    for (unsigned int i = 0; i < query.size(); i++) {
        tmp_query[i] = (float)query.data()[i];
    }
    indices.resize(knn * query.cols());
    distance2.resize(knn * query.cols());
    index->search(query.cols(), tmp_query.data(), knn, distance2.data(),
                  indices.data());

    return knn;
}

std::pair<core::Tensor, core::Tensor> 
KnnFaiss::SearchKNN_Tensor(const core::Tensor &query, 
                           int knn) const {
    core::SizeVector size = query.GetShape();
    size_t query_dim = size[1];
    size_t query_size = size[0];

    if (dataset_size_ <= 0 || query_dim != dimension_ || knn < 0) {
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }

    if (query.GetDtype() != core::Dtype::Float32){
        utility::LogWarning("[KnnFaiss::SearchKNN_Tensor] Unsupported Data type of Tensor.");
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }
    
    if(!support_on_gpu_ &&
        query.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA){
        utility::LogWarning("[KnnFaiss::SearchKNN_Tensor] Current Faiss Index is not supported on gpu. Query is moved to CPU memory.");
        core::Tensor query = query.Copy(core::Device("CPU:0"));
    }
    
    float *_data_ptr = static_cast<float *>(query.GetBlob()->GetDataPtr());

    std::vector<long> indices;
    std::vector<float> distance2;
    indices.resize(knn * query_size);
    distance2.resize(knn * query_size);
    index->search(query_size, _data_ptr, knn, distance2.data(),
                  indices.data());
    
    core::Tensor result_indices_(indices, {(long int)query_size, knn}, core::Dtype::Int32, query.GetBlob()->GetDevice());
    core::Tensor result_distance2_(distance2, {(long int)query_size, knn}, core::Dtype::Float32, query.GetBlob()->GetDevice());
    std::pair<core::Tensor, core::Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

template <typename T>
int KnnFaiss::SearchRadius(const T &query,
                           float radius,
                           std::vector<long> &indices,
                           std::vector<float> &distance2) const {
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_) {
        return -1;
    }
    std::vector<float> tmp_query(query.size());
    for (unsigned int i = 0; i < query.size(); i++) {
        tmp_query[i] = (float)query.data()[i];
    }
    faiss::RangeSearchResult result(query.cols());
    result.do_allocation();
    index->range_search(
            query.cols(), tmp_query.data(), std::pow(radius, 2),
            &result);  // square radius to maintain unify with kdtreeflann

    std::vector<long> tmp_indices;
    std::vector<float> tmp_distances;
    for (unsigned int i = 0; i < result.lims[query.cols()]; i++) {
        tmp_indices.push_back(result.labels[i]);
        tmp_distances.push_back(result.distances[i]);
    }

    // Sort indices & distances
    indices.resize(tmp_indices.size());
    distance2.resize(tmp_distances.size());
    std::vector<std::size_t> p(tmp_indices.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) {
        return tmp_distances[i] < tmp_distances[j];
    });
    std::transform(p.begin(), p.end(), indices.begin(),
                   [&](std::size_t i) { return tmp_indices[i]; });
    std::transform(p.begin(), p.end(), distance2.begin(),
                   [&](std::size_t i) { return tmp_distances[i]; });
    return result.lims[1];  // for just one query point, lims[1] == # of results
}

std::pair<core::Tensor, core::Tensor> 
KnnFaiss::SearchHybrid_Tensor(const core::Tensor &query,
                              float radius,
                              int max_knn) const{
    core::SizeVector size = query.GetShape();
    size_t query_dim = size[1];
    size_t query_size = size[0];

    if (dataset_size_ <= 0 || query_dim != dimension_ || max_knn < 0) {
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }

    if (query.GetDtype() != core::Dtype::Float32){
        utility::LogWarning("[KnnFaiss::SearchHybrid_Tensor] Unsupported Data type of Tensor.");
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }
    
    if(!support_on_gpu_ &&
        query.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA){
        utility::LogWarning("[KnnFaiss::SearchHybrid_Tensor] Current Faiss Index is not supported on gpu. Query is moved to CPU memory.");
        core::Tensor query = query.Copy(core::Device("CPU:0"));
    }
    
    float *_data_ptr = static_cast<float *>(query.GetBlob()->GetDataPtr());

    std::vector<long> indices;
    std::vector<float> distance2;
    indices.resize(max_knn * query_size);
    distance2.resize(max_knn * query_size);
    index->search(query_size, _data_ptr, max_knn, distance2.data(),
                  indices.data());
    
    unsigned int upper_ = max_knn * query_size;
    for (unsigned int i = 0; i < upper_; i++){
        if (distance2[i] < radius) {
            distance2[i] = 0;
            indices[i] = -1;
        }
    }

    core::Tensor result_indices_(indices, {(long int)query_size, max_knn}, core::Dtype::Int32, query.GetBlob()->GetDevice());
    core::Tensor result_distance2_(distance2, {(long int)query_size, max_knn}, core::Dtype::Float32, query.GetBlob()->GetDevice());
    std::pair<core::Tensor, core::Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

bool KnnFaiss::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();

    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[KnnFaiss::SetRawData] Failed due to no data.");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    for (unsigned int i = 0; i < dimension_ * dataset_size_; i++) {
        data_[i] = (float)data.data()[i];
    }
    index.reset(new faiss::IndexFlatL2(dimension_));
    index->add(dataset_size_, data_.data());
    return true;
}

template int KnnFaiss::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;
template int KnnFaiss::SearchKNN<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        int knn,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;
template int KnnFaiss::SearchRadius<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        float radius,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;

template int KnnFaiss::Search<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        const KDTreeSearchParam &param,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;
template int KnnFaiss::SearchKNN<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        int knn,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;
template int KnnFaiss::SearchRadius<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        float radius,
        std::vector<long> &indices,
        std::vector<float> &distance2) const;

}  // namespace geometry
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
