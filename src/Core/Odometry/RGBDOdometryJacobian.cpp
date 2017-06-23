// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

#include "Odometry.h"

#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Odometry/RGBDOdometryJacobian.h>

namespace three {

namespace {

const double SOBEL_SCALE = 0.125;
const double LAMBDA_HYBRID_DEPTH = 0.968;

}	// unnamed namespace

std::tuple<Eigen::Matrix6d, Eigen::Vector6d>
		RGBDOdometryJacobianFromColorTerm::ComputeJacobianAndResidual(
		const RGBDImage &source, const RGBDImage &target,
		const Image &source_xyz,
		const RGBDImage &target_dx, const RGBDImage &target_dy,
		const Eigen::Matrix3d &intrinsic,
		const Eigen::Matrix4d &extrinsic,
		const CorrespondenceSetPixelWise &corresps) const
{
	const double fx = intrinsic(0, 0);
	const double fy = intrinsic(1, 1);
	Eigen::Matrix3d R = extrinsic.block<3, 3>(0, 0);
	Eigen::Vector3d t = extrinsic.block<3, 1>(0, 3);

	Eigen::Matrix6d JTJ;
	Eigen::Vector6d JTr;
	JTJ.setZero();
	JTr.setZero();
	double res = 0.0;

#ifdef _OPENMP
#pragma omp parallel
{
#endif
	Eigen::Matrix6d JTJ_private;
	Eigen::Vector6d JTr_private;
	JTJ_private.setZero();
	JTr_private.setZero();
	double res_private = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int row = 0; row < corresps.size(); row++) {
		int u_s = corresps[row](0);
		int v_s = corresps[row](1);
		int u_t = corresps[row](2);
		int v_t = corresps[row](3);
		double diff = *PointerAt<float>(target.color_, u_t, v_t) -
				*PointerAt<float>(source.color_, u_s, v_s);
		double dIdx = SOBEL_SCALE * (*PointerAt<float>(target_dx.color_, u_t, v_t));
		double dIdy = SOBEL_SCALE * (*PointerAt<float>(target_dy.color_, u_t, v_t));
		Eigen::Vector3d p3d_mat(
				*PointerAt<float>(source_xyz, u_s, v_s, 0),
				*PointerAt<float>(source_xyz, u_s, v_s, 1),
				*PointerAt<float>(source_xyz, u_s, v_s, 2));
		Eigen::Vector3d p3d_trans = R * p3d_mat + t;
		double invz = 1. / p3d_trans(2);
		double c0 = dIdx * fx * invz;
		double c1 = dIdy * fy * invz;
		double c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;
		Eigen::Vector6d J_r;
		double r = diff;
		J_r(0) = -p3d_trans(2) * c1 + p3d_trans(1) * c2;
		J_r(1) = p3d_trans(2) * c0 - p3d_trans(0) * c2;
		J_r(2) = -p3d_trans(1) * c0 + p3d_trans(0) * c1;
		J_r(3) = c0;
		J_r(4) = c1;
		J_r(5) = c2;
		JTJ_private += J_r * J_r.transpose();
		JTr_private += J_r * r;
		res_private += diff * diff;
	}
#ifdef _OPENMP
#pragma omp critical
{
#endif
	JTJ += JTJ_private;
	JTr += JTr_private;
	res += res_private;
#ifdef _OPENMP
}	//	omp critical
}	//	omp parallel
#endif		
	res /= (double)corresps.size();
	PrintDebug("Residual : %.2e (# of points : %d)\n", res, corresps.size());

	return std::make_tuple(std::move(JTJ), std::move(JTr));
}

std::tuple<Eigen::Matrix6d, Eigen::Vector6d>
		RGBDOdometryJacobianFromHybridTerm::ComputeJacobianAndResidual(
		const RGBDImage &source, const RGBDImage &target,
		const Image &source_xyz,
		const RGBDImage &target_dx, const RGBDImage &target_dy,
		const Eigen::Matrix3d &intrinsic,
		const Eigen::Matrix4d &extrinsic,
		const CorrespondenceSetPixelWise &corresps) const
{
	double sqrt_lamba_dep, sqrt_lambda_img;
	sqrt_lamba_dep = sqrt(LAMBDA_HYBRID_DEPTH);
	sqrt_lambda_img = sqrt(1.0 - LAMBDA_HYBRID_DEPTH);

	const double fx = intrinsic(0, 0);
	const double fy = intrinsic(1, 1);
	Eigen::Matrix3d R = extrinsic.block<3, 3>(0, 0);
	Eigen::Vector3d t = extrinsic.block<3, 1>(0, 3);

	Eigen::Matrix6d JTJ;
	Eigen::Vector6d JTr;
	JTJ.setZero();
	JTr.setZero();
	double res_photo = 0.0;
	double res_geo = 0.0;

#ifdef _OPENMP
#pragma omp parallel
{
#endif	
	Eigen::Matrix6d JTJ_private;
	Eigen::Vector6d JTr_private;
	JTJ_private.setZero();
	JTr_private.setZero();
	double res_photo_private = 0.0;
	double res_geo_private = 0.0;
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (int row = 0; row < corresps.size(); row++) {
		int u_s = corresps[row](0);
		int v_s = corresps[row](1);
		int u_t = corresps[row](2);
		int v_t = corresps[row](3);
		double diff_photo = (*PointerAt<float>(target.color_, u_t, v_t) -
				*PointerAt<float>(source.color_, u_s, v_s));
		double dIdx = SOBEL_SCALE *
				(*PointerAt<float>(target_dx.color_, u_t, v_t));
		double dIdy = SOBEL_SCALE *
				(*PointerAt<float>(target_dy.color_, u_t, v_t));
		double dDdx = SOBEL_SCALE *
				(*PointerAt<float>(target_dx.depth_, u_t, v_t));
		double dDdy = SOBEL_SCALE *
				(*PointerAt<float>(target_dy.depth_, u_t, v_t));
		if (std::isnan(dDdx)) dDdx = 0;
		if (std::isnan(dDdy)) dDdy = 0;
		Eigen::Vector3d p3d_mat(
				*PointerAt<float>(source_xyz, u_s, v_s, 0),
				*PointerAt<float>(source_xyz, u_s, v_s, 1),
				*PointerAt<float>(source_xyz, u_s, v_s, 2));
		Eigen::Vector3d p3d_trans = R * p3d_mat + t;

		double diff_geo = *PointerAt<float>(target.depth_, u_t, v_t) -
				p3d_trans(2);
		double invz = 1. / p3d_trans(2);
		double c0 = dIdx * fx * invz;
		double c1 = dIdy * fy * invz;
		double c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;
		double d0 = dDdx * fx * invz;
		double d1 = dDdy * fy * invz;
		double d2 = -(d0 * p3d_trans(0) + d1 * p3d_trans(1)) * invz;
		
		Eigen::Vector6d J_r_photo;
		double r_photo = sqrt_lambda_img * diff_photo;
		J_r_photo(0) = sqrt_lambda_img * (-p3d_trans(2) * c1 + p3d_trans(1) * c2);
		J_r_photo(1) = sqrt_lambda_img * (p3d_trans(2) * c0 - p3d_trans(0) * c2);
		J_r_photo(2) = sqrt_lambda_img * (-p3d_trans(1) * c0 + p3d_trans(0) * c1);
		J_r_photo(3) = sqrt_lambda_img * (c0);
		J_r_photo(4) = sqrt_lambda_img * (c1);
		J_r_photo(5) = sqrt_lambda_img * (c2);
		JTJ_private += J_r_photo * J_r_photo.transpose();
		JTr_private += J_r_photo * r_photo;		
		res_photo_private += diff_photo * diff_photo;

		Eigen::Vector6d J_r_geo;
		double r_geo = sqrt_lamba_dep * diff_geo;
		J_r_geo(0) = sqrt_lamba_dep *
				((-p3d_trans(2) * d1 + p3d_trans(1) * d2) - p3d_trans(1));
		J_r_geo(1) = sqrt_lamba_dep *
				((p3d_trans(2) * d0 - p3d_trans(0) * d2) + p3d_trans(0));
		J_r_geo(2) = sqrt_lamba_dep *
				((-p3d_trans(1) * d0 + p3d_trans(0) * d1));
		J_r_geo(3) = sqrt_lamba_dep * (d0);
		J_r_geo(4) = sqrt_lamba_dep * (d1);
		J_r_geo(5) = sqrt_lamba_dep * (d2 - 1.0f);
		JTJ_private += J_r_geo * J_r_geo.transpose();
		JTr_private += J_r_geo * r_geo;
		res_geo_private += diff_geo * diff_geo;
	}
#ifdef _OPENMP
#pragma omp critical
{
#endif
	JTJ += JTJ_private;
	JTr += JTr_private;
	res_photo += res_photo_private;
	res_geo += res_geo_private;
#ifdef _OPENMP
}	//	omp critical
}	//	omp parallel
#endif		
	res_photo /= (double)corresps.size();
	res_geo /= (double)corresps.size();
	PrintDebug("Residual : %.2e (photometric) + %.2e (geometric) (# of points : %d)\n",
			res_photo, res_geo, corresps.size());

	return std::make_tuple(std::move(JTJ), std::move(JTr));
}

}	// namespace three
