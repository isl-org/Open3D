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

#include <cstdlib>
#include <ctime>

#include <Core/Utility/Console.h>
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)source.points_.size(); i++) {
		std::vector<int> indices(1);
		std::vector<double> dists(1);
		const auto &point = source.points_[i];
		if (target_kdtree.SearchHybrid(point, max_correspondence_distance, 1,
				indices, dists) > 0) {
#ifdef _OPENMP
#pragma omp critical
#endif
			{
				corres.push_back(Eigen::Vector2i(i, indices[0]));
				error2 += dists[0];
			}
		}
	}
	if (corres.empty()) {
		result.fitness = 0.0;
		result.inlier_rmse = 0.0;
	} else {
		result.fitness = (double)corres.size() / (double)source.points_.size();
		result.inlier_rmse = std::sqrt(error2 / (double)corres.size());
	}
}

void EvaluateRANSAC(const PointCloud &source, const PointCloud &target,
		const CorrespondenceSet &corres, double max_correspondence_distance,
		RegistrationResult &result)
{
	double error2 = 0.0;
	int good = 0;
	double max_dis2 = max_correspondence_distance * max_correspondence_distance;
	for (const auto &c : corres) {
		double dis2 =
				(source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
		if (dis2 < max_dis2) {
			good++;
			error2 += dis2;
		}
	}
	if (good == 0) {
		result.fitness = 0.0;
		result.inlier_rmse = 0.0;
	} else {
		result.fitness = (double)good / (double)corres.size();
		result.inlier_rmse = std::sqrt(error2 / (double)good);
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
	result.inlier_rmse = 0.0;
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
				result.fitness, result.inlier_rmse);
		Eigen::Matrix4d update = estimation.ComputeTransformation(
				pcd, target, corres);
		RegistrationResult backup = result;
		result.transformation = update * result.transformation;
		pcd.Transform(update);
		GetICPCorrespondence(pcd, target, kdtree, max_correspondence_distance,
				corres, result);
		if (std::abs(backup.inlier_rmse - result.inlier_rmse) < 
				criteria.relative_rmse_) {
			break;
		}
	}
	return result;
}

RegistrationResult RegistrationRANSAC(const PointCloud &source,
		const PointCloud &target, const CorrespondenceSet &corres,
		double max_correspondence_distance,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		int ransac_n/* = 6*/, int max_ransac_iteration/* = 1000*/)
{
	RegistrationResult result;
	result.transformation = Eigen::Matrix4d::Identity();
	result.inlier_rmse = 0.0;
	result.fitness = 0.0;
	if (ransac_n < 3 || (int)corres.size() < ransac_n ||
			max_correspondence_distance <= 0.0) {
		return result;
	}
	std::srand((unsigned int)std::time(0));
	CorrespondenceSet ransac_corres(ransac_n);
	CorrespondenceSet icp_corres;
	for (int i = 0; i < max_ransac_iteration; i++) {
		RegistrationResult this_result;
		for (int j = 0; j < ransac_n; j++) {
			ransac_corres[j] = corres[std::rand() % (int)corres.size()];
		}
		this_result.transformation = estimation.ComputeTransformation(source,
				target, ransac_corres);
		PointCloud pcd = source;
		pcd.Transform(this_result.transformation);
		EvaluateRANSAC(pcd, target, corres, max_correspondence_distance,
				this_result);
		if (this_result.fitness > result.fitness ||
				(this_result.fitness == result.fitness &&
				this_result.inlier_rmse < result.inlier_rmse)) {
			result = this_result;
		}
	}
	PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness,
			result.inlier_rmse);
	return result;
}

}	// namespace three
