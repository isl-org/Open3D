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
		data_ = ((const PointCloud &)geometry).points_;
		dimension_ = 3;
		break;
	case Geometry::GEOMETRY_TRIANGLEMESH:
		data_ = ((const TriangleMesh &)geometry).vertices_;
		dimension_ = 3;
		break;
	case Geometry::GEOMETRY_IMAGE:
	case Geometry::GEOMETRY_UNKNOWN:
	default:
		return false;
		break;
	}
	if (data_.size() == 0) {
		return false;
	}
	flann_dataset_.reset(new flann::Matrix<double>((double *)data_.data(),
			data_.size(), dimension_));
	dataset_size_ = data_.size();
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
	if (data_.empty() || dataset_size_ <= 0 || query.rows() != dimension_) {
		return -1;
	}
	if (max_nn < 0) {
		max_nn = (int)dataset_size_;
	}
	flann::Matrix<double> query_flann((double *)query.data(), 1, dimension_);
	flann::SearchParams param(-1, 0.0);
	param.max_neighbors = max_nn;
	indices.resize(max_nn);
	distance2.resize(max_nn);
	flann::Matrix<int> indices_flann(indices.data(), query_flann.rows, max_nn);
	flann::Matrix<double> dists_flann(distance2.data(), query_flann.rows,
			max_nn);
	return flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
			float(radius * radius), param);
}

template int KDTreeFlann::Search<Eigen::Vector3d>(const Eigen::Vector3d &query,
		const three::KDTreeSearchParam &param, std::vector<int> &indices,
		std::vector<double> &distance2);
template int KDTreeFlann::SearchKNN<Eigen::Vector3d>(
		const Eigen::Vector3d &query, int knn, std::vector<int> &indices,
		std::vector<double> &distance2);
template int KDTreeFlann::SearchRadius<Eigen::Vector3d>(
		const Eigen::Vector3d &query, double radius, std::vector<int> &indices,
		std::vector<double> &distance2, int max_nn = -1);

}	// namespace three
