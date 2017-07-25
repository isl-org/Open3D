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

#include "PointCloudForColoredICP.h"

#include <iostream>
#include <Eigen/Dense>
#include <Core/Registration/Registration.h>
#include <Core/Registration/TransformationEstimation.h>
//#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace three{

//void PointCloudForColoredICP::Clear()
//{
//	points_.clear();
//	normals_.clear();
//	colors_.clear();
//}
	
void MyInitializeGradient(PointCloudForColoredICP &target,
		double radius_for_gradient_computation)
{
	//std::cout << "radius_for_gradient_computation" << std::endl;
	//std::cout << radius_for_gradient_computation << std::endl;

	//three::KDTreeFlann tree;
	//tree.SetGeometry(target);
	//
	//target.color_gradient_.resize(target.points_.size());
	//for (size_t k = 0; k < target.points_.size(); k++) {
	//	Eigen::Vector3d temp = Eigen::Vector3d::Zero();
	//	target.color_gradient_[k] = temp;
	//}

	//for (size_t k = 0; k < target.points_.size(); k++) {
	//	Eigen::Vector3d vt = target.points_[k];
	//	Eigen::Vector3d nt = target.normals_[k];
	//	const double it = (target.colors_[k](0) + target.colors_[k](1) 
	//			+ target.colors_[k](2)) / 3.0f;
	//																						
	//	std::vector< int > pointIdx;
	//	std::vector< double > pointSquaredDistance;

	//	if (tree.SearchHybrid(vt, radius_for_gradient_computation,
	//			100, pointIdx, pointSquaredDistance) > 0) {
	//		// approximate image gradient of vt's tangential plane
	//		size_t nn = pointIdx.size();
	//		if (nn >= 3) {
	//			Eigen::MatrixXd A(nn, 3);
	//			Eigen::MatrixXd b(nn, 1);	
	//			A.setZero();
	//			b.setZero();
	//			for (int i = 1; i < nn; i++) {
	//				int P_adj_idx = pointIdx[i];
	//				Eigen::Vector3d vt_adj = target.points_[P_adj_idx];
	//				Eigen::Vector3d vt_proj = 
	//						vt_adj - (vt_adj - vt).dot(nt) * nt;
	//				double it_adj = (target.colors_[P_adj_idx](0) 
	//						+ target.colors_[P_adj_idx](1) 
	//						+ target.colors_[P_adj_idx](2)) / 3.0f;
	//				A(i - 1, 0) = (vt_proj(0) - vt(0));
	//				A(i - 1, 1) = (vt_proj(1) - vt(1));
	//				A(i - 1, 2) = (vt_proj(2) - vt(2));
	//				b(i - 1, 0) = (it_adj - it);
	//			}			
	//			// adds orthogonal constraint
	//			A(nn - 1, 0) = (nn - 1) * nt(0);
	//			A(nn - 1, 1) = (nn - 1) * nt(1);
	//			A(nn - 1, 2) = (nn - 1) * nt(2);
	//			b(nn - 1, 0) = 0;
	//			// solving linear equation
	//			bool is_success;
	//			Eigen::MatrixXd x;
	//			std::tie(is_success, x) = SolveLinearSystem(
	//				A.transpose() * A, A.transpose() * b);
	//			if (is_success) {
	//				target.color_gradient_[k](0) = x(0, 0);
	//				target.color_gradient_[k](1) = x(1, 0);
	//				target.color_gradient_[k](2) = x(2, 0);
	//			} else {
	//				target.color_gradient_[k](0) = 0.0f;
	//				target.color_gradient_[k](1) = 0.0f;
	//				target.color_gradient_[k](2) = 0.0f;
	//			}
	//		} else {
	//			target.color_gradient_[k](0) = 0.0f;
	//			target.color_gradient_[k](1) = 0.0f;
	//			target.color_gradient_[k](2) = 0.0f;
	//		}
	//	}
	//}
}

Eigen::Matrix4d MyComputeTransformation(
		const PointCloudForColoredICP &source, 
		PointCloudForColoredICP &target,
		const CorrespondenceSet &corres,
		double radius_for_gradient_computation,
		double lambda_geometric_,
		double lambda_photometric_)
{
	//if (corres.empty() || target.HasNormals() == false ||
	//	target.HasColors() == false || source.HasColors() == false)
	//	return Eigen::Matrix4d::Identity();
	//if (target.color_gradient_.size() != target.points_.size())
	//	MyInitializeGradient(target, radius_for_gradient_computation);

	//double sqrt_lambda_geometric_ = sqrt(lambda_geometric_);
	//double sqrt_lambda_photometric_ = sqrt(lambda_photometric_);

	//auto compute_jacobian_and_residual = [&]
	//		(int i, std::vector<Eigen::Vector6d> &J_r, std::vector<double> &r) {

	//	const size_t cs = corres[i][0];
	//	const size_t ct = corres[i][1];
	//	const Eigen::Vector3d &vs = source.points_[cs];
	//	const Eigen::Vector3d &vt = target.points_[ct];
	//	const Eigen::Vector3d &nt = target.normals_[ct];		

	//	J_r.resize(2);
	//	r.resize(2);
	//	
	//	J_r[0].block<3, 1>(0, 0) = sqrt_lambda_geometric_ * vs.cross(nt);
	//	J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric_ * nt;
	//	r[0] = sqrt_lambda_geometric_ * (vs - vt).dot(nt);

	//	// project vs into vt's tangential plane
	//	const Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;
	//	const double is = (source.colors_[cs](0) + source.colors_[cs](1) 
	//			+ source.colors_[cs](2)) / 3.0f;
	//	const double it = (target.colors_[ct](0) + target.colors_[ct](1)
	//			+ target.colors_[ct](2)) / 3.0f;
	//	const Eigen::Vector3d dit = target.color_gradient_[ct];		
	//	const double is0_proj = (dit.dot(vs_proj - vt)) + it;
	//	//std::cout << dit.transpose() << std::endl;
	//	//std::cout << is0_proj << std::endl;
	//	
	//	Eigen::Matrix3d M;
	//	M(0, 0) = 1 - nt(0) * nt(0);
	//	M(0, 1) = -nt(0) * nt(1);
	//	M(0, 2) = -nt(0) * nt(2);
	//	M(1, 0) = -nt(0) * nt(1);
	//	M(1, 1) = 1 - nt(1) * nt(1);
	//	M(1, 2) = -nt(1) * nt(2);
	//	M(2, 0) = -nt(0) * nt(2);
	//	M(2, 1) = -nt(1) * nt(2);
	//	M(2, 2) = 1 - nt(2) * nt(2);

	//	Eigen::Vector3d ditM;
	//	ditM = dit.transpose() * M;
	//	J_r[1].block<3, 1>(0, 0) = sqrt_lambda_photometric_ * vs.cross(ditM);
	//	J_r[1].block<3, 1>(3, 0) = sqrt_lambda_photometric_ * ditM;
	//	if (i == 10000)
	//		std::cout << J_r[0] << std::endl;
	//	r[1] = sqrt_lambda_photometric_ * (is - is0_proj);
	//};

	//Eigen::Matrix6d JTJ;
	//Eigen::Vector6d JTr;
	//std::tie(JTJ, JTr) = ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
	//		compute_jacobian_and_residual, (int)corres.size());

	//bool is_success;
	//Eigen::Matrix4d extrinsic;
	//std::tie(is_success, extrinsic) = 
	//		SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

	//return is_success ? extrinsic : Eigen::Matrix4d::Identity();
	return Eigen::Matrix4d::Identity();

}

}	// namespace three
