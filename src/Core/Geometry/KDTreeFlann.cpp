// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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
#pragma warning(disable: 4267)
#endif

#include "KDTreeFlann.h"

#include <flann/flann.hpp>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/TriangleMesh.h>

namespace three{

KDTreeFlann::KDTreeFlann()
{
}

KDTreeFlann::~KDTreeFlann()
{
}

bool KDTreeFlann::SetGeometry(const Geometry &geometry)
{
	switch (geometry.GetGeometryType()) {
	case Geometry::GEOMETRY_POINTCLOUD:
		dataset_size_ = ((const PointCloud &)geometry).points_.size();
		data_.resize(dataset_size_ * 3);
		memcpy(data_.data(), ((const PointCloud &)geometry).points_.data(),
				dataset_size_ * 3 * sizeof(double));
		dimension_ = 3;
		break;
	case Geometry::GEOMETRY_TRIANGLEMESH:
		dataset_size_ = ((const TriangleMesh &)geometry).vertices_.size();
		data_.resize(dataset_size_ * 3);
		memcpy(data_.data(), ((const TriangleMesh &)geometry).vertices_.data(),
				dataset_size_ * 3 * sizeof(double));
		dimension_ = 3;
		break;
	case Geometry::GEOMETRY_IMAGE:
	case Geometry::GEOMETRY_UNKNOWN:
	default:
		return false;
		break;
	}
	if (dataset_size_ == 0) {
		return false;
	}
	flann_dataset_.reset(new flann::Matrix<double>((double *)data_.data(),
			dataset_size_, dimension_));
	flann_index_.reset(new flann::Index<flann::L2<double>>(*flann_dataset_,
		flann::KDTreeSingleIndexParams(15)));
	flann_index_->buildIndex();
	return true;
}

template<typename T>
int KDTreeFlann::Search(const T &query, const KDTreeSearchParam &param, 
			std::vector<int> &indices, std::vector<double> &distance2)
{
	switch (param.GetSearchType()) {
	case KDTreeSearchParam::SEARCH_KNN:
		return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_, 
				indices, distance2);
	case KDTreeSearchParam::SEARCH_RADIUS:
		return SearchRadius(query, 
				((const KDTreeSearchParamRadius &)param).radius_, indices, 
				distance2, ((const KDTreeSearchParamRadius &)param).max_nn_);
	default:
		return -1;
	}
	return -1;
}

template<typename T>
int KDTreeFlann::SearchKNN(const T &query, int knn, std::vector<int> &indices,
		std::vector<double> &distance2)
{
	// This is optimized code for heavily repeated search
	// Other flann::Index::knnSearch() implementations lose performance due to
	// memory allocation/deallocation
	if (data_.empty() || dataset_size_ <= 0 || 
			query.rows() != dimension_ || knn < 0)
	{
		return -1;
	}
	flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
	indices.resize(knn);
	distance2.resize(knn);
	flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, knn);
	flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows, knn);
	return flann_index_->knnSearch(query_flann, indices_flann, dists_flann, knn,
			flann::SearchParams(-1, 0.0));
}

template<typename T>
int KDTreeFlann::SearchRadius(const T &query, double radius,
		std::vector<int> &indices, std::vector<double> &distance2,
		int max_nn/* = -1*/)
{
	// This is optimized code for heavily repeated search
	// Other flann::Index::radiusSearch() implementations lose performance due
	// to memory management and CPU caching
	if (data_.empty() || dataset_size_ <= 0 || query.rows() != dimension_) {
		return -1;
	}
	flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
	flann::SearchParams param(-1, 0.0);
	param.max_neighbors = max_nn;
	std::vector<std::vector<int>> indices_vec(1);
	std::vector<std::vector<double>> dists_vec(1);
	int k = flann_index_->radiusSearch(query_flann, indices_vec, dists_vec,
			float(radius * radius), param);
	indices = indices_vec[0];
	distance2 = dists_vec[0];
	return k;
}

template int KDTreeFlann::Search<Eigen::Vector3d>(const Eigen::Vector3d &query,
		const three::KDTreeSearchParam &param, std::vector<int> &indices,
		std::vector<double> &distance2);
template int KDTreeFlann::SearchKNN<Eigen::Vector3d>(
		const Eigen::Vector3d &query, int knn, std::vector<int> &indices,
		std::vector<double> &distance2);
template int KDTreeFlann::SearchRadius<Eigen::Vector3d>(
		const Eigen::Vector3d &query, double radius, std::vector<int> &indices,
		std::vector<double> &distance2, int max_nn);

}	// namespace three

#ifdef _MSC_VER
#pragma warning(pop)
#endif
