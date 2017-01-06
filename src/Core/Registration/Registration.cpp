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

#include "Registration.h"

#include <Core/Utility/Console.h>
#include <Core/Utility/Timer.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace three {

namespace {

void GetICPCorrespondence(const PointCloud &source, const PointCloud &target,
		const KDTreeFlann &target_kdtree, double max_correspondence_distance,
		CorrespondenceSet &corres, RegistrationResult &result)
{
	corres.clear();
	double error2 = 0.0;
	std::vector<int> indices;
	std::vector<double> dists;
	for (size_t i = 0; i < source.points_.size(); i++) {
		const auto &point = source.points_[i];
		int k = target_kdtree.SearchRadius(point, max_correspondence_distance,
				indices, dists);
		if (k > 0) {
			corres.push_back(std::make_pair((int)i, indices[0]));
			error2 += dists[0];
		}
	}
	if (corres.empty()) {
		result.fitness = 0.0;
		result.rmse = 0.0;
	} else {
		result.fitness = (double)corres.size() / (double)source.points_.size();
		result.rmse = std::sqrt(error2 / (double)corres.size());
	}
}

}	// unnamed namespace

RegistrationResult RegistrationICP(const PointCloud &source,
		const PointCloud &target, double max_correspondence_distance,
		const Eigen::Matrix4d &init/* = Eigen::Matrix4d::Identity()*/,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		const ConvergenceCriteria &criteria/* = ConvergenceCriteria()*/)
{
	RegistrationResult result;
	result.transformation = init;
	result.rmse = 0.0;
	result.fitness = 0.0;
	if (max_correspondence_distance <= 0.0) {
		return result;
	}
	KDTreeFlann kdtree;
	kdtree.SetGeometry(target);
	PointCloud pcd = source;
	if (init.isIdentity() == false) {
		pcd.Transform(init);
	}
	CorrespondenceSet corres;
	GetICPCorrespondence(pcd, target, kdtree, max_correspondence_distance,
			corres, result);
	for (int i = 0; i < criteria.max_iteration_; i++) {
		PrintDebug("ICP Iteration #%d: Fitness %.4f, RMSE %.4f\n", i,
				result.fitness, result.rmse);
		Eigen::Matrix4d update = estimation.ComputeTransformation(
				pcd, target, corres);
		RegistrationResult backup = result;
		result.transformation = update * result.transformation;
		pcd.Transform(update);
		GetICPCorrespondence(pcd, target, kdtree, max_correspondence_distance,
				corres, result);
		if (std::abs(backup.rmse - result.rmse) < criteria.relative_rmse_) {
			break;
		}
	}
	return result;
}

}	// namespace three
