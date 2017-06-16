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
#include <Core/Registration/Feature.h>

namespace three {

namespace {

std::tuple<RegistrationResult, CorrespondenceSet>
		GetRegistrationResultAndCorrespondences(
		const PointCloud &source, const PointCloud &target,
		const KDTreeFlann &target_kdtree, double max_correspondence_distance,
		const Eigen::Matrix4d &transformation)
{
	RegistrationResult result(transformation);
	CorrespondenceSet corres;
	if (max_correspondence_distance <= 0.0) {
		return std::make_tuple(std::move(result), std::move(corres));
	}
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
		result.fitness_ = 0.0;
		result.inlier_rmse_ = 0.0;
	} else {
		result.fitness_ = (double)corres.size() / (double)source.points_.size();
		result.inlier_rmse_ = std::sqrt(error2 / (double)corres.size());
	}
	return std::make_tuple(std::move(result), std::move(corres));
}

RegistrationResult EvaluateRANSACBasedOnCorrespondence(const PointCloud &source,
		const PointCloud &target, const CorrespondenceSet &corres,
		double max_correspondence_distance,
		const Eigen::Matrix4d &transformation)
{
	RegistrationResult result(transformation);
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
		result.fitness_ = 0.0;
		result.inlier_rmse_ = 0.0;
	} else {
		result.fitness_ = (double)good / (double)corres.size();
		result.inlier_rmse_ = std::sqrt(error2 / (double)good);
	}
	return result;
}

}	// unnamed namespace

RegistrationResult EvaluateRegistration(const PointCloud &source,
		const PointCloud &target, double max_correspondence_distance,
		const Eigen::Matrix4d &transformation/* = Eigen::Matrix4d::Identity()*/)
{
	KDTreeFlann kdtree;
	kdtree.SetGeometry(target);
	PointCloud pcd = source;
	if (transformation.isIdentity() == false) {
		pcd.Transform(transformation);
	}
	return std::get<0>(GetRegistrationResultAndCorrespondences(pcd, target,
			kdtree, max_correspondence_distance, transformation));
}

RegistrationResult RegistrationICP(const PointCloud &source,
		const PointCloud &target, double max_correspondence_distance,
		const Eigen::Matrix4d &init/* = Eigen::Matrix4d::Identity()*/,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		const ICPConvergenceCriteria &criteria/* = ICPConvergenceCriteria()*/)
{
	if (max_correspondence_distance <= 0.0) {
		return RegistrationResult(init);
	}
	Eigen::Matrix4d transformation = init;
	KDTreeFlann kdtree;
	kdtree.SetGeometry(target);
	PointCloud pcd = source;
	if (init.isIdentity() == false) {
		pcd.Transform(init);
	}
	RegistrationResult result;
	CorrespondenceSet corres;
	std::tie(result, corres) = GetRegistrationResultAndCorrespondences(
			pcd, target, kdtree, max_correspondence_distance, transformation);
	for (int i = 0; i < criteria.max_iteration_; i++) {
		PrintDebug("ICP Iteration #%d: Fitness %.4f, RMSE %.4f\n", i,
				result.fitness_, result.inlier_rmse_);
		Eigen::Matrix4d update = estimation.ComputeTransformation(
				pcd, target, corres);
		transformation = update * transformation;
		pcd.Transform(update);
		RegistrationResult backup = result;
		std::tie(result, corres) = GetRegistrationResultAndCorrespondences(pcd,
				target, kdtree, max_correspondence_distance, transformation);
		if (std::abs(backup.fitness_ - result.fitness_) <
				criteria.relative_fitness_ && std::abs(backup.inlier_rmse_ -
				result.inlier_rmse_) < criteria.relative_rmse_) {
			break;
		}
	}
	return result;
}

RegistrationResult RegistrationRANSACBasedOnCorrespondence(
		const PointCloud &source, const PointCloud &target,
		const CorrespondenceSet &corres, double max_correspondence_distance,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		int ransac_n/* = 6*/, const RANSACConvergenceCriteria &criteria
		/* = RANSACConvergenceCriteria()*/)
{
	if (ransac_n < 3 || (int)corres.size() < ransac_n ||
			max_correspondence_distance <= 0.0) {
		return RegistrationResult();
	}
	std::srand((unsigned int)std::time(0));
	Eigen::Matrix4d transformation;
	CorrespondenceSet ransac_corres(ransac_n);
	RegistrationResult result;
	for (int itr = 0; itr < criteria.max_iteration_ &&
			itr < criteria.max_validation_; itr++) {
		for (int j = 0; j < ransac_n; j++) {
			ransac_corres[j] = corres[std::rand() % (int)corres.size()];
		}
		transformation = estimation.ComputeTransformation(source,
				target, ransac_corres);
		PointCloud pcd = source;
		pcd.Transform(transformation);
		auto this_result = EvaluateRANSACBasedOnCorrespondence(pcd, target,
				corres, max_correspondence_distance, transformation);
		if (this_result.fitness_ > result.fitness_ ||
				(this_result.fitness_ == result.fitness_ &&
				this_result.inlier_rmse_ < result.inlier_rmse_)) {
			result = this_result;
		}
	}
	PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness_,
			result.inlier_rmse_);
	return result;
}

RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
		const PointCloud &source, const PointCloud &target,
		const Feature &source_feature, const Feature &target_feature,
		double max_correspondence_distance,
		const TransformationEstimation &estimation
		/* = TransformationEstimationPointToPoint(false)*/,
		int ransac_n/* = 4*/, const std::vector<std::reference_wrapper<const
		CorrespondenceChecker>> &checkers/* = {}*/,
		const RANSACConvergenceCriteria &criteria
		/* = RANSACConvergenceCriteria()*/)
{
	if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
		return RegistrationResult();
	}
	std::srand((unsigned int)std::time(0));
	RegistrationResult result;
	CorrespondenceSet ransac_corres(ransac_n);
	KDTreeFlann kdtree(target);
	KDTreeFlann kdtree_feature(target_feature);
	std::vector<int> indices(1);
	std::vector<double> dists(1);
	for (int itr = 0, val = 0; itr < criteria.max_iteration_ &&
			val < criteria.max_validation_; itr++) {
		Eigen::Matrix4d transformation;
		for (int j = 0; j < ransac_n; j++) {
			ransac_corres[j](0) = std::rand() % (int)source.points_.size();
			if (kdtree_feature.SearchKNN(Eigen::VectorXd(
					source_feature.data_.col(ransac_corres[j](0))), 1, indices,
					dists) == 0) {
				PrintDebug("[RegistrationRANSACBasedOnFeatureMatching] Found a feature without neighbors.\n");
				ransac_corres[j](1) = 0;
			} else {
				ransac_corres[j](1) = indices[0];
			}
		}
		for (const auto &checker : checkers) {
			if (checker.get().require_pointcloud_alignment_ == false &&
					checker.get().Check(source, target, ransac_corres,
					transformation) == false) {
				continue;
			}
		}
		transformation = estimation.ComputeTransformation(source, target,
				ransac_corres);
		for (const auto &checker : checkers) {
			if (checker.get().require_pointcloud_alignment_ == true &&
					checker.get().Check(source, target, ransac_corres,
					transformation) == false) {
				continue;
			}
		}
		PointCloud pcd = source;
		pcd.Transform(transformation);
		auto this_result = std::get<0>(GetRegistrationResultAndCorrespondences(
				pcd, target, kdtree, max_correspondence_distance,
				transformation));
		if (this_result.fitness_ > result.fitness_ ||
				(this_result.fitness_ == result.fitness_ &&
				this_result.inlier_rmse_ < result.inlier_rmse_)) {
			result = this_result;
		}
		val++;
	}
	PrintDebug("RANSAC: Fitness %.4f, RMSE %.4f\n", result.fitness_,
			result.inlier_rmse_);
	return result;
}

}	// namespace three
