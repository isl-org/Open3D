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

#include "PointCloud.h"

namespace three{

namespace {

std::pair<Eigen::Vector3d, double> ComputeNormalAndCurvature(
		const PointCloud &cloud, const std::vector<int> &indices)
{
	Eigen::Vector3d normal;
	double curvature = 0.0;
	if (indices.size() == 0) {
		return std::make_pair(normal, curvature);
	}
	Eigen::Matrix3d covariance;
	Eigen::Matrix<double, 9, 1> cumulants;
	for (size_t i = 0; i < indices.size(); i++) {
		const Eigen::Vector3d &point = cloud.points_[indices[i]];
		cumulants(0) += point(0);
		cumulants(1) += point(1);
		cumulants(2) += point(2);
		cumulants(3) += point(0) * point(0);
		cumulants(4) += point(0) * point(1);
		cumulants(5) += point(0) * point(2);
		cumulants(6) += point(1) * point(1);
		cumulants(7) += point(1) * point(2);
		cumulants(8) += point(2) * point(2);
	}
	cumulants /= (double)indices.size();
	covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
	covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
	covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
	covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
	covariance(1, 0) = covariance(0, 1);
	covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
	covariance(2, 0) = covariance(0, 2);
	covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
	covariance(2, 1) = covariance(1, 2);

	// TODO

	return std::make_pair(normal, curvature);
}

}	// unnamed namespace

bool EstimateNormalsAndCurvatures(PointCloud &cloud,
		const KDTreeSearchParam &search_param/* = KDTreeSearchParamKNN()*/)
{
	bool has_normal = cloud.HasNormals();
	if (cloud.HasNormals() == false) {
		cloud.normals_.resize(cloud.points_.size());
	}
	if (cloud.HasCurvatures() == false) {
		cloud.curvatures_.resize(cloud.points_.size());
	}
	// TODO
	// https://github.com/PointCloudLibrary/pcl/blob/a654fe4188382416c99322cafbd9319c59a7355c/common/include/pcl/common/impl/centroid.hpp
	// computeMeanAndCovarianceMatrix 
	return true;
}

bool EstimateNormalsAndCurvatures(PointCloud &cloud,
		const Eigen::Vector3d &orientation_reference,
		const KDTreeSearchParam &search_param/* = KDTreeSearchParamKNN()*/)
{
	if (cloud.HasNormals() == false) {
		cloud.normals_.resize(cloud.points_.size());
	}
	if (cloud.HasCurvatures() == false) {
		cloud.curvatures_.resize(cloud.points_.size());
	}
	return true;
}

}	// namespace three
