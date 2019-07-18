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

#include "Open3D/Geometry/KDTreeFlann.h"

#include <flann/flann.hpp>

#include "Open3D/Geometry/HalfEdgeTriangleMesh.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const Eigen::MatrixXd &data) { SetMatrixData(data); }

KDTreeFlann::KDTreeFlann(const Geometry &geometry) { SetGeometry(geometry); }

KDTreeFlann::KDTreeFlann(const registration::Feature &feature) {
    SetFeature(feature);
}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KDTreeFlann::SetGeometry(const Geometry &geometry) {
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
                    "[KDTreeFlann::SetGeometry] Unsupported Geometry type.\n");
            return false;
    }
}

bool KDTreeFlann::SetFeature(const registration::Feature &feature) {
    return SetMatrixData(feature.data_);
}

template <typename T>
int KDTreeFlann::Search(const T &query,
                        const KDTreeSearchParam &param,
                        std::vector<int> &indices,
                        std::vector<double> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
            return SearchHybrid(
                    query, ((const KDTreeSearchParamHybrid &)param).radius_,
                    ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           std::vector<int> &indices,
                           std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }
    flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
    indices.resize(knn);
    distance2.resize(knn);
    flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, knn);
    flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows, knn);
    int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                    knn, flann::SearchParams(-1, 0.0));
    indices.resize(k);
    distance2.resize(k);
    return k;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              double radius,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_) {
        return -1;
    }
    flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = -1;
    std::vector<std::vector<int>> indices_vec(1);
    std::vector<std::vector<double>> dists_vec(1);
    int k = flann_index_->radiusSearch(query_flann, indices_vec, dists_vec,
                                       float(radius * radius), param);
    indices = indices_vec[0];
    distance2 = dists_vec[0];
    return k;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const T &query,
                              double radius,
                              int max_nn,
                              std::vector<int> &indices,
                              std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || max_nn < 0) {
        return -1;
    }
    flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    indices.resize(max_nn);
    distance2.resize(max_nn);
    flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, max_nn);
    flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows,
                                      max_nn);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    indices.resize(k);
    distance2.resize(k);
    return k;
}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning(
                "[KDTreeFlann::SetRawData] Failed due to no data.\n");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    memcpy(data_.data(), data.data(),
           dataset_size_ * dimension_ * sizeof(double));
    flann_dataset_.reset(new flann::Matrix<double>((double *)data_.data(),
                                                   dataset_size_, dimension_));
    flann_index_.reset(new flann::Index<flann::L2<double>>(
            *flann_dataset_, flann::KDTreeSingleIndexParams(15)));
    flann_index_->buildIndex();
    return true;
}

template int KDTreeFlann::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTreeFlann::Search<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

}  // namespace geometry
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif
