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

#pragma once

#include <memory>
#include <flann/flann.hpp>

#include "IGeometryOwner.h"
#include "KDTreeSearchParam.h"

namespace three {

class KDTreeFlann : public IGeometryOwner
{
public:
	KDTreeFlann() {}
	virtual ~KDTreeFlann() {}
	KDTreeFlann(const KDTreeFlann &) = delete;
	KDTreeFlann &operator=(const KDTreeFlann &) = delete;
	
public:
	// functions inherited from IGeometryOwner
	bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr) override;
	bool UpdateGeometry() override;
	bool HasGeometry() const override;

	template<typename T>
	int Search(const T &query, KDTreeSearchParam &param, 
			std::vector<int> &indices, std::vector<double> &distance2);

	template<typename T>
	int SearchKNN(const T &query, int knn, std::vector<int> &indices,
			std::vector<double> &distance2);

	template<typename T>
	int SearchRadius(const T &query, double radius, std::vector<int> &indices,
			std::vector<double> &distance2, int max_nn = -1);

protected:
	std::shared_ptr<const Geometry> geometry_ptr_;
	std::unique_ptr<flann::Matrix<double>> flann_dataset_;
	std::unique_ptr<flann::Index<flann::L2<double>>> flann_index_;
	int dimension_ = 0;
	size_t dataset_size_ = 0;
};

template<typename T>
int KDTreeFlann::Search(const T &query, KDTreeSearchParam &param, 
			std::vector<int> &indices, std::vector<double> &distance2)
{
	switch (param.GetSearchType()) {
	case KDTreeSearchParam::SEARCH_KNN:
		return SearchKNN(query, ((KDTreeSearchParamKNN &)param).knn_, 
				indices, distance2);
	case KDTreeSearchParam::SEARCH_RADIUS:
		return SearchRadius(query, ((KDTreeSearchParamRadius &)param).radius_, 
				indices, distance2, ((KDTreeSearchParamRadius &)param).max_nn_);
	default:
		return -1;
	}
	return -1;
}

template<typename T>
int KDTreeFlann::SearchKNN(const T &query, int knn, std::vector<int> &indices,
		std::vector<double> &distance2)
{
	if (HasGeometry() == false || query.rows() != dimension_ || knn < 0) {
		return -1;
	}
	flann::Matrix<double> query_flann((double *)query.data(), 1, 3);
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
	if (HasGeometry() == false || query.rows() != dimension_) {
		return -1;
	}
	if (max_nn < 0) {
		max_nn = (int)dataset_size_;
	}
	flann::Matrix<double> query_flann((double *)query.data(), 1, 3);
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

}	// namespace three
