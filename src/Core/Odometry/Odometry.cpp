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
#include <json/json.h>

namespace three {

namespace {

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
							double exist_d1 = double{
								*PointerAt<float>(depth1, exist_u1, exist_v1) }
								*(KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7]
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
		const OdometryOption& opt,
		const double& fx, const double& fy)
{
	Eigen::Matrix3d R = odo.block<3, 3>(0, 0);
	Eigen::Vector3d t = odo.block<3, 1>(0, 3);

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

	const double R_raw[9] = 
			{ R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2) };
	const double t_raw[3] = 
			{ t(0), t(1), t(2) };

	Eigen::Vector3d temp, p3d_mat, p3d_trans;
	double sobel_scale = opt.sobel_scale_;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int u1 = *PointerAt<int>(corresps, u0, v0, 0);
			int v1 = *PointerAt<int>(corresps, u0, v0, 1);
			if (u1 != -1 && v1 != -1) {

				double diff = double{ *PointerAt<float>(image1, u1, v1) -
					*PointerAt<float>(image0, u0, v0) };

				double dIdx = sobel_scale *
							double{ *PointerAt<float>(dI_dx1, u1, v1) };
				double dIdy = sobel_scale *
							double{ *PointerAt<float>(dI_dy1, u1, v1) };
				double dDdx = sobel_scale *
							double{ *PointerAt<float>(dD_dx1, u1, v1) };
				double dDdy = sobel_scale *
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

PinholeCameraIntrinsic ReadCameraIntrinsic(const std::string& intrinsic_path) {
	PinholeCameraIntrinsic intrinsic;
	if (intrinsic_path.empty() || 
			!ReadIJsonConvertible(intrinsic_path, intrinsic)) {
		PrintWarning("Failed to read intrinsic parameters for depth image.\n");
		PrintWarning("Use default value for Primesense camera.\n");
		intrinsic.SetIntrinsics(640, 480, 525.0, 525.0, 319.5, 239.5);
	}
	return intrinsic;
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

void PreprocessDepth(const three::Image &depth, const OdometryOption& opt)
{
	for (int y = 0; y < depth.height_; y++) {
		for (int x = 0; x < depth.width_; x++) {
			float *p = PointerAt<float>(depth, x, y);
			if ((*p > opt.max_depth_ || *p < opt.min_depth_ || *p <= 0))
				*p = std::numeric_limits<float>::quiet_NaN();
		}
	}
}

bool CheckImagePair(const three::Image &color0, const three::Image &depth0,
		const three::Image &color1, const three::Image &depth1) {
	return (color0.width_ == color1.width_ && color0.height_ == color1.height_
			&& depth0.width_ == depth1.width_ && depth0.height_ == depth1.height_
			&& color0.width_ == depth0.width_ && color0.height_ == depth0.height_
			&& color1.width_ == depth1.width_ && color1.height_ == depth1.height_
			&& color0.bytes_per_channel_ == 1 && color1.bytes_per_channel_ == 1
			&& depth0.bytes_per_channel_ == 2 && depth1.bytes_per_channel_ == 2);
}

Eigen::Matrix4d Transform6DVecorto4x4Matrix(Eigen::VectorXd input) {
	Eigen::Affine3d aff_mat;
	aff_mat.linear() = (Eigen::Matrix3d)
		Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ())
		* Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX());
	aff_mat.translation() = Eigen::Vector3d(input(3), input(4), input(5));
	return aff_mat.matrix();
}

std::tuple<std::shared_ptr<Image>, std::shared_ptr<Image>, 
		std::shared_ptr<Image>, std::shared_ptr<Image>> InitializeRGBDPair(
				const Image& color0_8bit, const Image& depth0_16bit,
				const Image& color1_8bit, const Image& depth1_16bit,
				PinholeCameraIntrinsic& camera_intrinsic,
				Eigen::Matrix4d& odo_init,
				const OdometryOption& opt) 
{
	auto gray_temp0 = CreateFloatImageFromImage(color0_8bit);
	auto gray_temp1 = CreateFloatImageFromImage(color1_8bit);
	auto gray0 = FilterImage(*gray_temp0, FILTER_GAUSSIAN_3);
	auto gray1 = FilterImage(*gray_temp1, FILTER_GAUSSIAN_3);

	double depth_scale = opt.is_tum_ ? 5000.0 : 1000.0;
	double max_depth = opt.max_depth_;
	auto depth_temp0 = ConvertDepthToFloatImage(depth0_16bit, depth_scale, max_depth);
	auto depth_temp1 = ConvertDepthToFloatImage(depth1_16bit, depth_scale, max_depth);
	PreprocessDepth(*depth_temp0, opt);
	PreprocessDepth(*depth_temp1, opt);
	auto depth0 = FilterImage(*depth_temp0, FILTER_GAUSSIAN_3);
	auto depth1 = FilterImage(*depth_temp1, FILTER_GAUSSIAN_3);

	std::shared_ptr<Image> temp_corresps;
	int corresps_count;
	std::tie(temp_corresps, corresps_count) = ComputeCorrespondence(
			camera_intrinsic.intrinsic_matrix_, odo_init.inverse(), 
			*depth0, *depth1, opt);
	PrintDebug("Number of correspondence is %d\n", corresps_count);

	if (opt.fast_reject_ && corresps_count < opt.minimum_corr_) {
		PrintWarning("[InitializeRGBDPair] Bad initial pose\n");
	}
	NormalizeIntensity(*gray0, *gray1, *temp_corresps);
	return std::make_tuple(gray0, depth0, gray1, depth1);
}

std::tuple<bool, Eigen::Matrix4d> ComputeOdometryMultiscale(
		const Eigen::Matrix4d& init_odo,
		const Image &gray0, const Image &depth0,
		const Image &gray1, const Image &depth1,
		const double lambda_dep,
		const PinholeCameraIntrinsic& camera_intrinsic,
		const OdometryOption& opt) {

	std::vector<int> iter_counts;
	for (int i = 0; i < opt.num_pyramid_; i++)
		iter_counts.push_back(opt.num_iter_);

	auto pyramid_gray0 = CreateImagePyramid(gray0, opt.num_pyramid_);
	auto pyramid_gray1 = CreateImagePyramid(gray1, opt.num_pyramid_);
	auto pyramid_depth0 = CreateImagePyramid(depth0, opt.num_pyramid_, false);
	auto pyramid_depth1 = CreateImagePyramid(depth1, opt.num_pyramid_, false);
	auto pyramid_dI_dx1 = FilterImagePyramid(pyramid_gray1, FILTER_SOBEL_3_DX);
	auto pyramid_dI_dy1 = FilterImagePyramid(pyramid_gray1, FILTER_SOBEL_3_DY);
	auto pyramid_dD_dx1 = FilterImagePyramid(pyramid_depth1, FILTER_SOBEL_3_DX);
	auto pyramid_dD_dy1 = FilterImagePyramid(pyramid_depth1, FILTER_SOBEL_3_DY);

	Eigen::Matrix4d result_odo = init_odo.isZero() ?
			Eigen::Matrix4d::Identity() : init_odo;
	Eigen::Matrix4d curr_odo;
	Eigen::VectorXd ksi;
	int corresps_count;

	std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
			CreateCameraMatrixPyramid(camera_intrinsic, (int)iter_counts.size());

	bool is_success = true;
	for (int level = (int)iter_counts.size() - 1; level >= 0; level--) {
		const Eigen::Matrix3d level_camera_matrix = pyramid_camera_matrix[level];

		auto level_cloud0 = ConvertDepth2Cloud(
			*pyramid_depth0[level], level_camera_matrix);
		const double fx = level_camera_matrix(0, 0);
		const double fy = level_camera_matrix(1, 1);

		for (int iter = 0; iter < iter_counts[level]; iter++) {
			PrintDebug("Iter : %d, Level : %d, ", iter, level);

			std::shared_ptr<Image> corresps;
			int corresps_count;
			std::tie(corresps, corresps_count) = ComputeCorrespondence(
					level_camera_matrix, result_odo.inverse(),
					*pyramid_depth0[level], *pyramid_depth1[level], opt);

			if (corresps_count == 0) {
				PrintError("[ComputeOdometry] Num of corres is 0!\n");
				is_success = false;
			}

			bool solutionExist;
			Eigen::VectorXd ksi;
			std::tie(solutionExist, ksi) = ComputeKsi(
					*pyramid_gray0[level], *level_cloud0, *pyramid_gray1[level],
					*pyramid_dI_dx1[level], *pyramid_dI_dy1[level],
					*pyramid_depth0[level], *pyramid_depth1[level],
					*pyramid_dD_dx1[level], *pyramid_dD_dy1[level],
					result_odo, *corresps, corresps_count,
					opt, fx, fy);

			if (!solutionExist) {
				PrintError("[ComputeOdometry] no solution!\n");
				is_success = false;
			}

			curr_odo = Transform6DVecorto4x4Matrix(ksi);
			result_odo = curr_odo * result_odo;
		}
	}
	
	if (corresps_count < opt.minimum_corr_)
		is_success = false;

	return std::make_tuple(is_success, result_odo);
}

}	// unnamed namespace

std::tuple<bool, Eigen::Matrix4d, Eigen::MatrixXd> 
		ComputeRGBDOdometry(
		const Image& color0_8bit, const Image& depth0_16bit,
		const Image& color1_8bit, const Image& depth1_16bit,
		const OdometryOption& opt)
{
	if (!CheckImagePair(color0_8bit, depth0_16bit, color1_8bit, depth1_16bit)) {
		PrintError("[ComputeRGBDOdometry] Two RGBD pairs should be same in size.\n");
		PrintError("Color image should be 8bit and depth image should be 16bit\n");
		return std::make_tuple(false, 
				Eigen::Matrix4d::Identity(), Eigen::MatrixXd::Zero(6,6));
	}

	PinholeCameraIntrinsic camera_intrinsic = 
			ReadCameraIntrinsic(opt.intrinsic_path_);

	double lambda_dep = opt.lambda_dep_;
	if (lambda_dep < 0.0f || lambda_dep > 1.0f)
		lambda_dep = 0.95;
	PrintDebug("lambda_dep : %f\n", lambda_dep);
	PrintDebug("lambda_img : %f\n", 1.0f - lambda_dep);

	std::shared_ptr<Image> gray0, depth0, gray1, depth1;
	Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
	std::tie(gray0, depth0, gray1, depth1) = 
			InitializeRGBDPair(color0_8bit, depth0_16bit, 
					color1_8bit, depth1_16bit,
					camera_intrinsic, odo_init, opt);

	Eigen::Matrix4d odo;
	bool is_found;
	std::tie(is_found, odo) = ComputeOdometryMultiscale(odo_init,
			*gray0, *depth0, *gray1, *depth1,
			lambda_dep, camera_intrinsic, opt);

	Eigen::Matrix4d trans_output = odo.inverse();
	Eigen::MatrixXd info_output = CreateInfomationMatrix(
			odo, camera_intrinsic, *depth0, *depth1, opt);

	return std::make_tuple(true, trans_output, info_output);
}

}	// namespace three
