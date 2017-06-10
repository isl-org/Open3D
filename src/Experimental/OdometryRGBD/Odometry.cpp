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


#include <iostream> // this is just for debugging
#include "Odometry.h"

namespace three {

// needs discussion
void Odometry::PreprocessDepth(const three::Image &depth)
{
	//float *p = (float *)depth.data_.data();
	for (int y = 0; y < depth.height_; y++) {
		for (int x = 0; x < depth.width_; x++) {
			float *p = PointerAt<float>(depth, x, y);
			if ((*p > maxDepth || *p < minDepth || *p <= 0)) 
				*p = std::numeric_limits<float>::quiet_NaN();
		}
	}
}

std::shared_ptr<Image> Odometry::ConvertDepth2Cloud(
		const Image& depth, const Eigen::Matrix3d& camera_matrix)
{
	auto cloud = std::make_shared<Image>();
	if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
		PrintDebug("[ConvertDepth2Cloud] Unsupported image format.\n");
		return cloud;
	}
	const double inv_fx = 1.f / camera_matrix(0, 0);
	const double inv_fy = 1.f / camera_matrix(1, 1);
	const double ox = camera_matrix(0, 2);
	const double oy = camera_matrix(1, 2);
	cloud->PrepareImage(depth.width_, depth.height_, 3, 4); // xyz float type

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

inline void Odometry::set2shorts(int& dst, int short_v1, int short_v2)
{
	unsigned short* ptr = reinterpret_cast<unsigned short*>(&dst);
	ptr[0] = static_cast<unsigned short>(short_v1);
	ptr[1] = static_cast<unsigned short>(short_v2);
}

inline void Odometry::get2shorts(int src, int& short_v1, int& short_v2)
{
	typedef union { int vint32; unsigned short vuint16[2]; } s32tou16;
	const unsigned short* ptr = (reinterpret_cast<s32tou16*>(&src))->vuint16;
	short_v1 = ptr[0];
	short_v2 = ptr[1];
}

void Odometry::setconst(const Image& image, const int value) {
	for (int v = 0; v < image.height_; v++) {
		for (int u = 0; u < image.width_; u++) {
			*PointerAt<int>(image, u, v) = value;
		}
	}
}

int Odometry::ComputeCorrespondence(const Eigen::Matrix3d& K,
	const Eigen::Matrix4d& Rt,
	const Image& depth0, const Image& depth1, Image& corresps)
{
	corresps.PrepareImage(depth1.width_, depth1.height_, 1, 4);

	const Eigen::Matrix3d& K_inv = K.inverse();
	const Eigen::Matrix3d R = Rt.block<3, 3>(0, 0);
	const Eigen::Matrix3d KRK_inv = K * R * K_inv;
	const double KRK_inv_ptr[9] = 
			{ KRK_inv(0,0), KRK_inv(0,1), KRK_inv(0,2),
			KRK_inv(1,0), KRK_inv(1,1), KRK_inv(1,2),
			KRK_inv(2,0), KRK_inv(2,1), KRK_inv(2,2) };

	Eigen::Vector3d Kt = K * Rt.block<3, 1>(0, 3);
	const double Kt_ptr[3] = { Kt(0), Kt(1), Kt(2) };

	setconst(corresps, -1);
	int correspCount = 0;
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
						std::abs(transformed_d1 - d0) <= maxDepthDiff) {
						int* c = PointerAt<int>(corresps, u0, v0);
						if (*c != -1) {
							int exist_u1, exist_v1;
							get2shorts(*c, exist_u1, exist_v1);
							double exist_d1 = double{ 
									*PointerAt<float>(depth1, exist_u1, exist_v1) }
									* (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] 
									* exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2];
							if (transformed_d1 > exist_d1)
								continue;
						} else {
							correspCount++;
						}
						set2shorts(*c, u1, v1);
					}
				}
			}
		}
	}
	return correspCount;
}

bool Odometry::ComputeKsi(const Image& image0, const Image& cloud0,
	const Image& image1, const Image& dI_dx1, const Image& dI_dy1,
	const Image& depth0, const Image& depth1,
	const Image& dD_dx1, const Image& dD_dy1,
	const Eigen::Matrix4d& Rt,
	const Image& corresps, int corresps_count,
	const double& fx, const double& fy, const double& determinant_threshold,
	Eigen::VectorXd& ksi,
	int iter, int level)
{
	Eigen::Matrix3d R = Rt.block<3, 3>(0, 0);
	Eigen::Vector3d t = Rt.block<3, 1>(0, 3);

	int DoF = 6;
	Eigen::MatrixXd J(corresps_count * 2, DoF);
	Eigen::MatrixXd r(corresps_count * 2, 1);
	J.setConstant(0.f);
	r.setConstant(0.f);

	double res1 = 0.0;
	double res2 = 0.0;

	int point_count = 0;
	
	double sqrt_lamba_dep, sqrt_lambda_img;
	sqrt_lamba_dep = sqrt(LAMBDA_DEP_DEFAULT);
	sqrt_lambda_img = sqrt(1.0-LAMBDA_DEP_DEFAULT);

	const double R_raw[9] =
			{ R(0,0), R(0,1), R(0,2),
			R(1,0), R(1,1), R(1,2),
			R(2,0), R(2,1), R(2,2) };
	const double t_raw[3] =
			{ t(0), t(1), t(2) };

	Eigen::Vector3d temp, p3d_mat, p3d_trans;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int c = *PointerAt<int>(corresps, u0, v0);
			if (c != -1) {

				int u1, v1;
				get2shorts(c, u1, v1);

				double diff = static_cast<double>(*PointerAt<float>(image1, u1, v1)) -
						static_cast<double>(*PointerAt<float>(image0, u0, v0));

				double dIdx = sobelScale_i * double{ *PointerAt<float>(dI_dx1, u1, v1) };
				double dIdy = sobelScale_i * double{ *PointerAt<float>(dI_dy1, u1, v1) };
				double dDdx = sobelScale_d * double{ *PointerAt<float>(dD_dx1, u1, v1) };
				double dDdy = sobelScale_d * double{ *PointerAt<float>(dD_dy1, u1, v1) };
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

	if (verbose_) {
		PrintDebug("Res : %.2e + %.2e (Npts : %d)\n", res1, res2, point_count);
	}

	// solve system
	Eigen::MatrixXd Jt = J.transpose();
	Eigen::MatrixXd JtJ = Jt * J;
	Eigen::MatrixXd Jtr = Jt * r;

	bool solutionExist = true;
	double det = JtJ.determinant();
	if (fabs(det) < determinant_threshold || std::isnan(det) || std::isinf(det))
		solutionExist = false;

	if (solutionExist) {
		// Robust Cholesky decomposition of a matrix with pivoting.
		ksi = -JtJ.ldlt().solve(Jtr); 
	}	
	return solutionExist;
}

void Odometry::LoadCameraFile(const char* filename, Eigen::Matrix3d& K)
{
	if (strcmp(filename, "") == 0) {
		PrintDebug("[LoadCameraFile] Using default camera intrinsic");
		K << 525.0, 0, 319.5,
				0, 525.0, 239.5,
				0, 0, 1;
	}
	else {
		float fx_, fy_, cx_, cy_;
		float ICP_trunc_, integration_trunc_;

		FILE * f = fopen(filename, "r");
		if (f != NULL) {
			char buffer[1024];
			char* temp;
			// todo: fancy c++ style file read.
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%f", &fx_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &fy_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &cx_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &cy_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &ICP_trunc_);
					temp = fgets(buffer, 1024, f);
					sscanf(buffer, "%f", &integration_trunc_);
					temp = fgets(buffer, 1024, f);
				}
			}
			fclose(f);
		}
		K << fx_, 0.0, cx_,
				0.0, fy_, cy_,
				0.0, 0.0, 1.0;
	}
}

bool Odometry::Run(
	const Image& color0_8bit, const Image& depth0_16bit,
	const Image& color1_8bit, const Image& depth1_16bit,
	Eigen::Matrix4d& init_pose, 
	Eigen::Matrix4d& trans_output, Eigen::MatrixXd& info_output,
	const char* camera_filename,
	const double lambda_dep,
	bool verbose,
	bool fast_reject,
	bool is_tum)
{
	verbose_ = verbose;

	if (lambda_dep < 0.0f || lambda_dep > 1.0f)
		lambda_dep_ = LAMBDA_DEP_DEFAULT;
	else
		lambda_dep_ = lambda_dep;
	lambda_img_ = 1.0f - lambda_dep_;

	if (verbose) {
		PrintInfo("lambda_dep : %f\n", lambda_dep_);
		PrintInfo("lambda_img : %f\n", lambda_img_);
	}

	Eigen::Matrix3d camera_matrix;
	LoadCameraFile(camera_filename, camera_matrix);

	auto gray_temp0 = CreateFloatImageFromImage(color0_8bit);
	auto gray_temp1 = CreateFloatImageFromImage(color1_8bit);
	auto gray0 = FilterImage(*gray_temp0, FILTER_GAUSSIAN_3);
	auto gray1 = FilterImage(*gray_temp1, FILTER_GAUSSIAN_3);

	double depth_scale = is_tum ? 5000.0 : 1000.0;
	double max_depth = 4.0;
	auto depth0 = ConvertDepthToFloatImage(depth0_16bit, depth_scale, max_depth);
	auto depth1 = ConvertDepthToFloatImage(depth1_16bit, depth_scale, max_depth);
	PreprocessDepth(*depth0);
	PreprocessDepth(*depth1);
	auto depth_filtered_0 = FilterImage(*depth0, FILTER_GAUSSIAN_3);
	auto depth_filtered_1 = FilterImage(*depth1, FILTER_GAUSSIAN_3);

	Eigen::Matrix4d Rt_init = Eigen::Matrix4d::Identity();

	Image temp_corresps;
	int corresps_count = ComputeCorrespondence(
			camera_matrix, Rt_init.inverse(),
			*depth_filtered_0, *depth_filtered_1, temp_corresps);

	PrintDebug("Number of correspondence is %d\n", corresps_count);

	if (fast_reject) {
		if (corresps_count < MINIMUM_CORR)
			return false;
	}

	std::vector<int> iterCounts;
	for (int i = 0; i < NUM_PYRAMID; i++)
		iterCounts.push_back(NUM_ITER); 

	NormalizeIntensity(*gray0, *gray1, temp_corresps); 

	Eigen::Matrix4d Rt;
	bool isFound = ComputeOdometry(
			Rt, Rt_init,
			*gray0, *depth_filtered_0,
			*gray1, *depth_filtered_1,
			camera_matrix,
			iterCounts);

	Eigen::Matrix4d Rt_inv = Rt.inverse();
	trans_output = Rt_inv;
	info_output = CreateInfomationMatrix(Rt, camera_matrix, 
			*depth_filtered_0, *depth_filtered_1);

	return true;
}


Eigen::MatrixXd Odometry::CreateInfomationMatrix(
	const Eigen::Matrix4d& Rt, 
	const Eigen::Matrix3d& camera_matrix,
	const Image& depth0, const Image& depth1)
{
	Eigen::Matrix4d Rt_inv = Rt.inverse();

	Image corresps;
	int correspsCount = ComputeCorrespondence(camera_matrix, Rt_inv,
		depth0, depth1, corresps);
	
	auto point_cloud1 = ConvertDepth2Cloud(depth1, camera_matrix);

	// write q^*
	// see http://redwood-data.org/indoor/registration.html
	// note: I comes first and q_skew is scaled by factor 2.
	Eigen::MatrixXd GtG(6, 6);
	Eigen::MatrixXd G(3 * corresps.height_ * corresps.width_, 6);
	G.setConstant(0.0f);

	int cnt = 0;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int c = *PointerAt<int>(corresps, u0, v0);
			if (c != -1) {
				int u1, v1;
				get2shorts(c, u1, v1);	
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


void Odometry::NormalizeIntensity(
		Image& image0, Image& image1, Image& corresps)
{
	if (image0.width_ != image1.width_ ||
		image0.height_ != image1.height_) {
		PrintError("[NormalizeIntensity] Size of two input images should be same\n");
		return;
	}
	double point_count = 0.0, mean0 = 0.0, mean1 = 0.0;
	for (int v0 = 0; v0 < corresps.height_; v0++) {
		for (int u0 = 0; u0 < corresps.width_; u0++) {
			int c = *PointerAt<int>(corresps, u0, v0);
			if (c != -1) {
				int u1, v1;
				get2shorts(c, u1, v1);
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

std::vector<Eigen::Matrix3d> 
		Odometry::CreateCameraMatrixPyramid(Eigen::Matrix3d& K, int levels)
{
	std::vector<Eigen::Matrix3d> pyramid_camera_matrix;
	pyramid_camera_matrix.reserve(levels);
	for (int i = 0; i < levels; i++) {
		Eigen::Matrix3d level_camera_matrix = i == 0 ? 
				K : 0.5f * pyramid_camera_matrix[i - 1];
		level_camera_matrix(2, 2) = 1.;
		pyramid_camera_matrix.push_back(level_camera_matrix);
	}
	return pyramid_camera_matrix;
}

bool Odometry::ComputeOdometry(
		Eigen::Matrix4d& Rt, const Eigen::Matrix4d& initRt,
		const Image &gray0, const Image &depth0,
		const Image &gray1, const Image &depth1, 
		Eigen::Matrix3d& camera_matrix,
		const std::vector<int>& iterCounts) {

	if (((gray0.width_ != gray1.width_) || (gray1.height_ != gray1.height_)) ||
			((depth0.width_ != depth1.width_) || (depth0.height_ != depth1.height_)) ||
			((gray0.width_ != depth0.width_) || (gray0.height_ != depth0.height_)) ||
			((gray1.width_ != depth1.width_) || (gray1.height_ != depth1.height_))) {
		PrintError("[ComputeOdometry] Two RGBD pairs should be same in size.\n");
		return false;
	}

	auto pyramid_gray0 = CreateImagePyramid(gray0, NUM_PYRAMID);
	auto pyramid_gray1 = CreateImagePyramid(gray1, NUM_PYRAMID);
	auto pyramid_depth0 = CreateImagePyramid(depth0, NUM_PYRAMID, false);	
	auto pyramid_depth1 = CreateImagePyramid(depth1, NUM_PYRAMID, false);
	auto pyramid_dI_dx1 = FilterPyramidImage(pyramid_gray1, FILTER_SOBEL_3_DX);
	auto pyramid_dI_dy1 = FilterPyramidImage(pyramid_gray1, FILTER_SOBEL_3_DY);
	auto pyramid_dD_dx1 = FilterPyramidImage(pyramid_depth1, FILTER_SOBEL_3_DX);
	auto pyramid_dD_dy1 = FilterPyramidImage(pyramid_depth1, FILTER_SOBEL_3_DY);

	Eigen::Matrix4d resultRt = initRt.isZero() ? 
				Eigen::Matrix4d::Identity() : initRt;
	Eigen::Matrix4d currRt;
	Eigen::VectorXd ksi;
	int corresps_count;
	double res1, res2;

	std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
			CreateCameraMatrixPyramid(camera_matrix, (int)iterCounts.size());

	for (int level = (int)iterCounts.size() - 1; level >= 0; level--)
	{
		const Eigen::Matrix3d level_camera_matrix = pyramid_camera_matrix[level];

		auto level_cloud0 = ConvertDepth2Cloud(
				*pyramid_depth0[level], level_camera_matrix);
		const double fx = level_camera_matrix(0, 0);
		const double fy = level_camera_matrix(1, 1);
		const double determinant_threshold = 1e-6;

		for (int iter = 0; iter < iterCounts[level]; iter++) {
			if (verbose_) {
				PrintInfo("Iter : %d, Level : %d, ", iter, level);
			}

			Image corresps;
			corresps_count = ComputeCorrespondence(
					level_camera_matrix, resultRt.inverse(),
					*pyramid_depth0[level], *pyramid_depth1[level], corresps);

			if (corresps_count == 0)
			{
				if (verbose_) {
					PrintError("[ComputeOdometry] Num of corres is 0!\n");
				}
				break;
			}

			bool solutionExist = ComputeKsi(
				*pyramid_gray0[level], *level_cloud0, *pyramid_gray1[level], 
				*pyramid_dI_dx1[level], *pyramid_dI_dy1[level],
				*pyramid_depth0[level], *pyramid_depth1[level],
				*pyramid_dD_dx1[level], *pyramid_dD_dy1[level],
				resultRt, corresps, corresps_count,
				fx, fy, determinant_threshold,
				ksi,
				iter, level);

			if (!solutionExist)
			{
				if (verbose_) {
					PrintError("[ComputeOdometry] no solution!\n");
				}
				break;
			}

			Eigen::Affine3d aff_mat;
			aff_mat.linear() = (Eigen::Matrix3d) 
				Eigen::AngleAxisd(ksi(2), Eigen::Vector3d::UnitZ())
				* Eigen::AngleAxisd(ksi(1), Eigen::Vector3d::UnitY())
				* Eigen::AngleAxisd(ksi(0), Eigen::Vector3d::UnitX());
			aff_mat.translation() = Eigen::Vector3d(ksi(3), ksi(4), ksi(5));
			currRt = aff_mat.matrix();
			resultRt = currRt * resultRt;	
		}
	}
	Rt = resultRt;

	bool is_success = true;
	if (corresps_count < MINIMUM_CORR)
		is_success = false;

	return is_success;
}

}	// namespace three
