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

#include <Eigen/Dense>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>

namespace three {

namespace {

const double SOBEL_SCALE = 0.125;
const double LAMBDA_HYBRID_DEPTH = 0.968;

std::tuple<std::shared_ptr<Image>, int> 
		ComputeCorrespondence(
		const Eigen::Matrix3d intrinsic_matrix,
		const Eigen::Matrix4d& odo,
		const Image& depth0, const Image& depth1,
		const OdometryOption& opt)
{
	const Eigen::Matrix3d K = intrinsic_matrix;
	const Eigen::Matrix3d K_inv = K.inverse();
	const Eigen::Matrix3d R = odo.block<3, 3>(0, 0);
	const Eigen::Matrix3d KRK_inv = K * R * K_inv;
	const double KRK_inv_ptr[9] =
			{ KRK_inv(0,0), KRK_inv(0,1), KRK_inv(0,2),
			KRK_inv(1,0), KRK_inv(1,1), KRK_inv(1,2),
			KRK_inv(2,0), KRK_inv(2,1), KRK_inv(2,2) };

	Eigen::Vector3d Kt = K * odo.block<3, 1>(0, 3);
	const double Kt_ptr[3] = { Kt(0), Kt(1), Kt(2) };

	// initialization: filling with any (u,v) to (-1,-1)
	auto corresps = std::make_shared<Image>();
	corresps->PrepareImage(depth1.width_, depth1.height_, 2, 4);
	for (int v = 0; v < corresps->height_; v++) {
		for (int u = 0; u < corresps->width_; u++) {
			*PointerAt<long int>(*corresps, u, v, 0) = -1;
			*PointerAt<long int>(*corresps, u, v, 1) = -1;
		}
	}

	int corresp_count = 0;
	for (int v1 = 0; v1 < depth1.height_; v1++) {
		for (int u1 = 0; u1 < depth1.width_; u1++) {

			double d1 = double{ *PointerAt<float>(depth1, u1, v1) };
			if (!std::isnan(d1)) {
				double transformed_d1 = double{
					(d1 * (KRK_inv_ptr[6] * u1 + KRK_inv_ptr[7] *
					v1 + KRK_inv_ptr[8]) + Kt_ptr[2]) };
				int u0 = (int)(
						(d1 * (KRK_inv_ptr[0] * u1 + KRK_inv_ptr[1] *
						v1 + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1 + 0.5);
				int v0 = (int)(
						(d1 * (KRK_inv_ptr[3] * u1 + KRK_inv_ptr[4] *
						v1 + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1 + 0.5);

				if (u0 >= 0 && u0 < depth1.width_ &&
					v0 >= 0 && v0 < depth1.height_) {
					double d0 = double{ *PointerAt<float>(depth0, u0, v0) };
					if (!std::isnan(d0) &&
						std::abs(transformed_d1 - d0) <= opt.max_depth_diff_) {
						int exist_u1, exist_v1;
						exist_u1 = *PointerAt<int>(*corresps, u0, v0, 0);
						exist_v1 = *PointerAt<int>(*corresps, u0, v0, 1);
						if (exist_u1 != -1 && exist_v1 != -1) {
							double exist_d1 = double{ *PointerAt<float>
									(depth1, exist_u1, exist_v1) } * 
									(KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7]
									* exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2];
							if (transformed_d1 > exist_d1)
								continue;
						}
						else {
							corresp_count++;
						}
						*PointerAt<int>(*corresps, u0, v0, 0) = u1;
						*PointerAt<int>(*corresps, u0, v0, 1) = v1;
					}
				}
			}
		}
	}
	return std::make_tuple(corresps, corresp_count);
}

std::tuple<bool, Eigen::VectorXd> 
		ComputeKsi(const Image& image0, const Image& cloud0,
		const Image& image1, const Image& dI_dx1, const Image& dI_dy1,
		const Image& depth0, const Image& depth1,
		const Image& dD_dx1, const Image& dD_dy1,
		const Eigen::Matrix4d& odo,
		const Image& corresps, int corresps_count,
		const Eigen::Matrix3d& camera_matrix,
		const OdometryOption& opt)
{
	int DoF = 6;
	Eigen::MatrixXd J(corresps_count * 2, DoF);
	Eigen::MatrixXd r(corresps_count * 2, 1);
	J.setConstant(0.f);
	r.setConstant(0.f);

	double res1 = 0.0;
	double res2 = 0.0;

	int point_count = 0;

	double sqrt_lamba_dep, sqrt_lambda_img;
	sqrt_lamba_dep = sqrt(opt.lambda_dep_);
	sqrt_lambda_img = sqrt(1.0 - opt.lambda_dep_);

	const double fx = camera_matrix(0, 0);
	const double fy = camera_matrix(1, 1);

	Eigen::Matrix3d R = odo.block<3, 3>(0, 0);
	Eigen::Vector3d t = odo.block<3, 1>(0, 3);
	const double R_raw[9] = 
			{ R(0, 0), R(0, 1), R(0, 2),
			R(1, 0), R(1, 1), R(1, 2),
			R(2, 0), R(2, 1), R(2, 2) };
	const double t_raw[3] = 
			{ t(0), t(1), t(2) };

	Eigen::Vector3d temp, p3d_mat, p3d_trans;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int u1 = *PointerAt<int>(corresps, u0, v0, 0);
			int v1 = *PointerAt<int>(corresps, u0, v0, 1);
			if (u1 != -1 && v1 != -1) {

				double diff = double{ *PointerAt<float>(image1, u1, v1) -
						*PointerAt<float>(image0, u0, v0) };

				double dIdx = SOBEL_SCALE *
						double{ *PointerAt<float>(dI_dx1, u1, v1) };
				double dIdy = SOBEL_SCALE *
						double{ *PointerAt<float>(dI_dy1, u1, v1) };
				double dDdx = SOBEL_SCALE *
						double{ *PointerAt<float>(dD_dx1, u1, v1) };
				double dDdy = SOBEL_SCALE *
						double{ *PointerAt<float>(dD_dy1, u1, v1) };
				if (std::isnan(dDdx)) dDdx = 0;
				if (std::isnan(dDdy)) dDdy = 0;

				p3d_mat(0) = double{ *PointerAt<float>(cloud0, u0, v0, 0) };
				p3d_mat(1) = double{ *PointerAt<float>(cloud0, u0, v0, 1) };
				p3d_mat(2) = double{ *PointerAt<float>(cloud0, u0, v0, 2) };

				p3d_trans(0) = R_raw[0] * p3d_mat(0) + R_raw[1] * p3d_mat(1) +
						R_raw[2] * p3d_mat(2) + t_raw[0];
				p3d_trans(1) = R_raw[3] * p3d_mat(0) + R_raw[4] * p3d_mat(1) +
						R_raw[5] * p3d_mat(2) + t_raw[1];
				p3d_trans(2) = R_raw[6] * p3d_mat(0) + R_raw[7] * p3d_mat(1) +
						R_raw[8] * p3d_mat(2) + t_raw[2];

				double diff2 = double{ (*PointerAt<float>(depth1, u1, v1)) } -
					double{ (p3d_trans(2)) };

				double invz = 1. / p3d_trans(2);
				double c0 = dIdx * fx * invz;
				double c1 = dIdy * fy * invz;
				double c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;
				double d0 = dDdx * fx * invz;
				double d1 = dDdy * fy * invz;
				double d2 = -(d0 * p3d_trans(0) + d1 * p3d_trans(1)) * invz;

				int row1 = point_count * 2 + 0;
				int row2 = point_count * 2 + 1;
				J(row1, 0) = sqrt_lambda_img *
						(-p3d_trans(2) * c1 + p3d_trans(1) * c2);
				J(row1, 1) = sqrt_lambda_img *
						(p3d_trans(2) * c0 - p3d_trans(0) * c2);
				J(row1, 2) = sqrt_lambda_img *
						(-p3d_trans(1) * c0 + p3d_trans(0) * c1);
				J(row1, 3) = sqrt_lambda_img * (c0);
				J(row1, 4) = sqrt_lambda_img * (c1);
				J(row1, 5) = sqrt_lambda_img * (c2);
				r(row1, 0) = sqrt_lambda_img * diff;
				res1 += diff * diff;

				J(row2, 0) = sqrt_lamba_dep *
						((-p3d_trans(2) * d1 + p3d_trans(1) * d2) - p3d_trans(1));
				J(row2, 1) = sqrt_lamba_dep *
						((p3d_trans(2) * d0 - p3d_trans(0) * d2) + p3d_trans(0));
				J(row2, 2) = sqrt_lamba_dep *
						((-p3d_trans(1) * d0 + p3d_trans(0) * d1));
				J(row2, 3) = sqrt_lamba_dep * (d0);
				J(row2, 4) = sqrt_lamba_dep * (d1);
				J(row2, 5) = sqrt_lamba_dep * (d2 - 1.0f);
				r(row2, 0) = sqrt_lamba_dep * diff2;
				res2 += diff2 * diff2;

				point_count++;
			}
		}
	}
	res1 /= point_count;
	res2 /= point_count;

	PrintDebug("Res : %.2e + %.2e (Npts : %d)\n", res1, res2, point_count);

	Eigen::MatrixXd Jt = J.transpose();
	Eigen::MatrixXd JtJ = Jt * J;
	Eigen::MatrixXd Jtr = Jt * r;

	bool solutionExist = true;
	double det = JtJ.determinant();
	if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det))
		solutionExist = false;

	if (solutionExist) {
		// Robust Cholesky decomposition of a matrix with pivoting.
		Eigen::MatrixXd ksi = -JtJ.ldlt().solve(Jtr);
		return std::make_tuple(solutionExist, ksi);
	} else {
		return std::make_tuple(solutionExist, Eigen::VectorXd::Zero(6));
	}
}

std::shared_ptr<Image> ConvertDepth2Cloud(
		const Image& depth, const Eigen::Matrix3d& intrinsic_matrix)
{
	auto cloud = std::make_shared<Image>();
	if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
		PrintDebug("[ConvertDepth2Cloud] Unsupported image format.\n");
		return cloud;
	}
	const double inv_fx = 1.f / intrinsic_matrix(0, 0);
	const double inv_fy = 1.f / intrinsic_matrix(1, 1);
	const double ox = intrinsic_matrix(0, 2);
	const double oy = intrinsic_matrix(1, 2);
	cloud->PrepareImage(depth.width_, depth.height_, 3, 4);

	for (int y = 0; y < depth.height_; y++) {
		for (int x = 0; x < cloud->width_; x++) {
			float *px = PointerAt<float>(*cloud, x, y, 0);
			float *py = PointerAt<float>(*cloud, x, y, 1);
			float *pz = PointerAt<float>(*cloud, x, y, 2);
			float z = *PointerAt<float>(depth, x, y);
			*px = (float)((x - ox) * z * inv_fx);
			*py = (float)((y - oy) * z * inv_fy);
			*pz = z;
		}
	}
	return cloud;
}

std::vector<Eigen::Matrix3d>
		CreateCameraMatrixPyramid(
		const PinholeCameraIntrinsic& camera_intrinsic, int levels)
{
	std::vector<Eigen::Matrix3d> pyramid_camera_matrix;
	pyramid_camera_matrix.reserve(levels);
	for (int i = 0; i < levels; i++) {
		Eigen::Matrix3d level_camera_matrix = i == 0 ?
				camera_intrinsic.intrinsic_matrix_ : 
				0.5f * pyramid_camera_matrix[i - 1];
		level_camera_matrix(2, 2) = 1.;
		pyramid_camera_matrix.push_back(level_camera_matrix);
	}
	return pyramid_camera_matrix;
}

Eigen::MatrixXd CreateInfomationMatrix(
		const Eigen::Matrix4d& odo,
		const PinholeCameraIntrinsic& camera_intrinsic,
		const Image& depth0, const Image& depth1,
		const OdometryOption& opt)
{
	Eigen::Matrix4d odo_inv = odo.inverse();

	std::shared_ptr<Image> corresps; 
	int corresps_count;
	std::tie(corresps, corresps_count) = ComputeCorrespondence(
			camera_intrinsic.intrinsic_matrix_, odo_inv, depth0, depth1, opt);

	auto point_cloud1 = ConvertDepth2Cloud(
			depth1, camera_intrinsic.intrinsic_matrix_);

	// write q^*
	// see http://redwood-data.org/indoor/registration.html
	// note: I comes first and q_skew is scaled by factor 2.
	Eigen::MatrixXd GtG(6, 6);
	Eigen::MatrixXd G(3 * corresps->height_ * corresps->width_, 6);
	G.setConstant(0.0f);

	int cnt = 0;
	for (int v0 = 0; v0 < corresps->height_; v0++) {
		for (int u0 = 0; u0 < corresps->width_; u0++) {
			int u1 = *PointerAt<int>(*corresps, u0, v0, 0);
			int v1 = *PointerAt<int>(*corresps, u0, v0, 1);
			if (u1 != -1 && v1 != -1) {
				double x = double{ *PointerAt<float>(*point_cloud1, u1, v1, 0) };
				double y = double{ *PointerAt<float>(*point_cloud1, u1, v1, 1) };
				double z = double{ *PointerAt<float>(*point_cloud1, u1, v1, 2) };
				G(3 * cnt + 0, 0) = 1.0;
				G(3 * cnt + 0, 4) = 2.0 * z;
				G(3 * cnt + 0, 5) = -2.0 * y;
				G(3 * cnt + 1, 1) = 1.0;
				G(3 * cnt + 1, 3) = -2.0 * z;
				G(3 * cnt + 1, 5) = 2.0 * x;
				G(3 * cnt + 2, 2) = 1.0;
				G(3 * cnt + 2, 3) = 2.0 * y;
				G(3 * cnt + 2, 4) = -2.0 * x;
				cnt++;
			}
		}
	}
	GtG = G.transpose() * G;
	return GtG;
}

void NormalizeIntensity(Image& image0, Image& image1, Image& corresps) 
{
	if (image0.width_ != image1.width_ ||
		image0.height_ != image1.height_) {
		PrintError("[NormalizeIntensity] Size of two input images should be same\n");
		return;
	}
	double point_count = 0.0, mean0 = 0.0, mean1 = 0.0;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int u1 = *PointerAt<int>(corresps, u0, v0, 0);
			int v1 = *PointerAt<int>(corresps, u0, v0, 1);
			if (u1 != -1 && v1 != -1) {
				mean0 += *PointerAt<float>(image0, u0, v0);
				mean1 += *PointerAt<float>(image1, u1, v1);
				point_count++;
			}
		}
	}
	mean0 /= point_count;
	mean1 /= point_count;
	LinearTransformImage(image0, 0.5 / mean0, 0.0);
	LinearTransformImage(image1, 0.5 / mean1, 0.0);
}

inline std::shared_ptr<RGBDImage> PackRGBDImage(
	const Image &color, const Image &depth) {
	return std::make_shared<RGBDImage>(RGBDImage(color, depth));
}

std::shared_ptr<Image> PreprocessDepth(
		const Image &depth_orig, const OdometryOption& opt) 
{
	std::shared_ptr<Image> depth_processed = std::make_shared<Image>();
	*depth_processed = depth_orig;
	for (int y = 0; y < depth_processed->height_; y++) {
		for (int x = 0; x < depth_processed->width_; x++) {
			float *p = PointerAt<float>(*depth_processed, x, y);
			if ((*p < opt.min_depth_ || *p > opt.max_depth_ || *p <= 0))
				*p = std::numeric_limits<float>::quiet_NaN();			
		}
	}
	return depth_processed;
}

inline bool CheckImagePair(const Image& img1, const Image& img2) 
{
	return (img1.width_ == img2.width_ && img1.height_ == img2.height_);
}

inline bool CheckRGBDImagePair(const RGBDImage &source, const RGBDImage &target) 
{
	return (CheckImagePair(source.color_, target.color_) &&
			CheckImagePair(source.depth_, target.depth_) &&
			CheckImagePair(source.color_, source.depth_) &&
			CheckImagePair(target.color_, target.depth_) &&
			CheckImagePair(source.color_, target.color_) &&
			source.color_.bytes_per_channel_ == 4 && 
			target.color_.bytes_per_channel_ == 4 && 
			source.depth_.bytes_per_channel_ == 4 && 
			target.depth_.bytes_per_channel_ == 4);
}

Eigen::Matrix4d Transform6DVecorto4x4Matrix(Eigen::VectorXd input) 
{
	Eigen::Affine3d aff_mat;
	aff_mat.linear() = (Eigen::Matrix3d)
			Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ())
			* Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY())
			* Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX());
	aff_mat.translation() = Eigen::Vector3d(input(3), input(4), input(5));
	return aff_mat.matrix();
}

std::tuple<std::shared_ptr<RGBDImage>, std::shared_ptr<RGBDImage>>
		Initialization( 
		const RGBDImage& source_orig, const RGBDImage& target_orig,
		const PinholeCameraIntrinsic& camera_intrinsic,
		const Eigen::Matrix4d& odo_init,
		OdometryOption& opt) 
{
	if (opt.lambda_dep_ < 0.0f || opt.lambda_dep_ > 1.0f)
		opt.lambda_dep_ = LAMBDA_HYBRID_DEPTH;
	PrintDebug("lambda_dep : %f\n", opt.lambda_dep_);
	PrintDebug("lambda_img : %f\n", 1.0f - opt.lambda_dep_);

	auto source_gray = FilterImage(source_orig.color_, FILTER_GAUSSIAN_3);
	auto target_gray = FilterImage(target_orig.color_, FILTER_GAUSSIAN_3);
	auto source_depth_preprocessed = PreprocessDepth(source_orig.depth_, opt);
	auto target_depth_preprocessed = PreprocessDepth(target_orig.depth_, opt);
	auto source_depth = FilterImage(*source_depth_preprocessed, FILTER_GAUSSIAN_3);
	auto target_depth = FilterImage(*target_depth_preprocessed, FILTER_GAUSSIAN_3);

	std::shared_ptr<Image> temp_corresps;
	int corresps_count;
	std::tie(temp_corresps, corresps_count) = ComputeCorrespondence(
			camera_intrinsic.intrinsic_matrix_, odo_init.inverse(), 
			*source_depth, *target_depth, opt);
	PrintDebug("Number of correspondence is %d\n", corresps_count);

	int corresps_count_required = (int)(source_gray->height_ *
			source_gray->width_ * opt.minimum_correspondence_ratio_ + 0.5);
	if (opt.check_initialization_ && corresps_count < corresps_count_required) {
		PrintWarning("[InitializeRGBDPair] Bad initial pose\n");
	}
	NormalizeIntensity(*source_gray, *target_gray, *temp_corresps);

	auto source = PackRGBDImage(*source_gray, *source_depth);
	auto target = PackRGBDImage(*target_gray, *target_depth);
	return std::make_tuple(source, target);
}

std::tuple<bool, Eigen::Matrix4d> ComputeSingleIteration(
	const RGBDImage &source, const Image& source_cloud,
	const RGBDImage &target, 
	const RGBDImage &target_dx, const RGBDImage &target_dy,
	const Eigen::Matrix3d camera_matrix,
	const Eigen::Matrix4d& init_odo,
	const OdometryOption& opt)
{
	std::shared_ptr<Image> corresps;
	int corresps_count;
	std::tie(corresps, corresps_count) = ComputeCorrespondence(
			camera_matrix, init_odo.inverse(),
			source.depth_, target.depth_, opt);
	int corresps_count_required = (int)(source.color_.height_ * 
			source.color_.width_ * opt.minimum_correspondence_ratio_ + 0.5);
	if (corresps_count < corresps_count_required) {
		PrintError("[ComputeOdometry] %d is too fewer than mininum requirement %d\n",
				corresps_count, corresps_count_required);
		return std::make_tuple(false, Eigen::Matrix4d::Identity());
	}

	bool is_success;
	Eigen::VectorXd ksi;
	std::tie(is_success, ksi) = ComputeKsi(
			source.color_, source_cloud, target.color_,
			target_dx.color_, target_dy.color_,
			source.depth_, target.depth_,
			target_dx.depth_, target_dy.depth_,		
			init_odo, *corresps, corresps_count,
			camera_matrix, opt);

	if (!is_success) {
		PrintError("[ComputeOdometry] no solution!\n");
		return std::make_tuple(false, Eigen::Matrix4d::Identity());
	} else {
		return std::make_tuple(true, Transform6DVecorto4x4Matrix(ksi));
	}
}

std::tuple<bool, Eigen::Matrix4d> ComputeMultiscale(
		const RGBDImage &source, const RGBDImage &target,
		const PinholeCameraIntrinsic& camera_intrinsic,
		const Eigen::Matrix4d& init_odo,
		const OdometryOption& opt)
{
	std::vector<int> iter_counts = opt.iteration_number_per_pyramid_level_;
	int num_levels = (int)iter_counts.size();

	//auto source_pyramid = CreateRGBDImagePyramid(source, num_levels);
	//auto target_pyramid = CreateRGBDImagePyramid(target, num_levels);

	auto source_pyramid_gray = CreateImagePyramid(source.color_, num_levels);
	auto target_pyramid_gray = CreateImagePyramid(target.color_, num_levels);
	auto source_pyramid_depth = CreateImagePyramid(source.depth_, num_levels, 0);
	auto target_pyramid_depth = CreateImagePyramid(target.depth_, num_levels, 0);
	auto target_pyramid_gray_dx = 
			FilterImagePyramid(target_pyramid_gray, FILTER_SOBEL_3_DX);
	auto target_pyramid_gray_dy = 
			FilterImagePyramid(target_pyramid_gray, FILTER_SOBEL_3_DY);
	auto target_pyramid_depth_dx = 
			FilterImagePyramid(target_pyramid_depth, FILTER_SOBEL_3_DX);
	auto target_pyramid_depth_dy = 
			FilterImagePyramid(target_pyramid_depth, FILTER_SOBEL_3_DY);

	Eigen::Matrix4d result_odo = init_odo.isZero() ?
			Eigen::Matrix4d::Identity() : init_odo;	

	std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
			CreateCameraMatrixPyramid(camera_intrinsic, (int)iter_counts.size());

	for (int level = num_levels - 1; level >= 0; level--) {
		const Eigen::Matrix3d level_camera_matrix = pyramid_camera_matrix[level];

		auto source_cloud_level = ConvertDepth2Cloud(
				*source_pyramid_depth[level], level_camera_matrix);
		auto source_level = PackRGBDImage(*source_pyramid_gray[level],
				*source_pyramid_depth[level]);
		auto target_level = PackRGBDImage(*target_pyramid_gray[level], 
				*target_pyramid_depth[level]);
		auto target_dx_level = PackRGBDImage(*target_pyramid_gray_dx[level],
				*target_pyramid_depth_dx[level]);
		auto target_dy_level = PackRGBDImage(*target_pyramid_gray_dy[level],
				*target_pyramid_depth_dy[level]);
		
		for (int iter = 0; iter < iter_counts[num_levels - level - 1]; iter++) {
			PrintDebug("Iter : %d, Level : %d, ", iter, level);
			Eigen::Matrix4d curr_odo;
			bool is_success;
			std::tie(is_success, curr_odo) = ComputeSingleIteration(
				*source_level, *source_cloud_level, *target_level,
				*target_dx_level, *target_dy_level, level_camera_matrix,
				result_odo, opt);
			result_odo = curr_odo * result_odo;

			if (!is_success) {
				PrintError("[ComputeOdometry] no solution!\n");
				return std::make_tuple(false, Eigen::Matrix4d::Identity());
			}
		}
	}
	return std::make_tuple(true, result_odo);
}

}	// unnamed namespace

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> 
		ComputeRGBDOdometry(const RGBDImage &source, const RGBDImage &target,
		const PinholeCameraIntrinsic &camera_intrinsic /*=PinholeCameraIntrinsic()*/, 
		const Eigen::Matrix4d &odo_init /*= Eigen::Matrix4d::Identity()*/,
		OdometryOption &opt /*= OdometryOption()*/)
{
	if (!CheckRGBDImagePair(source, target)) {
		PrintError("[ComputeRGBDOdometry] Two RGBD pairs should be same in size.\n");
		return std::make_tuple(false, 
			Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Zero());
	}

	std::shared_ptr<RGBDImage> source_processed, target_processed;
	std::tie(source_processed, target_processed) =
			Initialization(source, target, camera_intrinsic, odo_init, opt);

	Eigen::Matrix4d odo;
	bool is_success;
	std::tie(is_success, odo) = ComputeMultiscale(
			*source_processed, *target_processed, 
			camera_intrinsic, odo_init, opt);

	if (is_success) {
		Eigen::Matrix4d trans_output = odo.inverse();
		Eigen::MatrixXd info_output = CreateInfomationMatrix(odo, camera_intrinsic,
			source_processed->depth_, target_processed->depth_, opt);
		return std::make_tuple(true, trans_output, info_output);
	} else {
		return std::make_tuple(false, 
				Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Zero());
	}
}

}	// namespace three
