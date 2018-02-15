// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include <Core/Odometry/RGBDOdometryJacobian.h>
#include <Core/Utility/Eigen.h>
#include <Core/Utility/Timer.h>

namespace three {

namespace {

std::tuple<std::shared_ptr<Image>, std::shared_ptr<Image>>
		InitializeCorrespondenceMap(int width, int height)
{
	// initialization: filling with any (u,v) to (-1,-1)
	auto correspondence_map = std::make_shared<Image>();
	auto depth_buffer = std::make_shared<Image>();
	correspondence_map->PrepareImage(width, height, 2, 4);
	depth_buffer->PrepareImage(width, height, 1, 4);
	for (int v = 0; v < correspondence_map->height_; v++) {
		for (int u = 0; u < correspondence_map->width_; u++) {
			*PointerAt<int>(*correspondence_map, u, v, 0) = -1;
			*PointerAt<int>(*correspondence_map, u, v, 1) = -1;
			*PointerAt<float>(*depth_buffer, u, v, 0) = -1.0f;
		}
	}
	return std::make_tuple(correspondence_map, depth_buffer);
}

inline void AddElementToCorrespondenceMap(
		Image &correspondence_map, Image &depth_buffer,
		int u_s, int v_s, int u_t, int v_t, float transformed_d_t) {
	int exist_u_t, exist_v_t;
	double exist_d_t;
	exist_u_t = *PointerAt<int>(correspondence_map, u_s, v_s, 0);
	exist_v_t = *PointerAt<int>(correspondence_map, u_s, v_s, 1);
	if (exist_u_t != -1 && exist_v_t != -1) {
		exist_d_t = *PointerAt<float>(depth_buffer, u_s, v_s);
		if (transformed_d_t < exist_d_t) { // update nearer point as correspondence
			*PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
			*PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
			*PointerAt<float>(depth_buffer, u_s, v_s) = transformed_d_t;
		}
	} else { // register correspondence
		*PointerAt<int>(correspondence_map, u_s, v_s, 0) = u_t;
		*PointerAt<int>(correspondence_map, u_s, v_s, 1) = v_t;
		*PointerAt<float>(depth_buffer, u_s, v_s) = transformed_d_t;
	}
}

void MergeCorrespondenceMaps(
		Image &correspondence_map, Image &depth_buffer,
		Image &correspondence_map_part, Image &depth_buffer_part)
{
	for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
		for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
			int u_t = *PointerAt<int>(correspondence_map_part, u_s, v_s, 0);
			int v_t = *PointerAt<int>(correspondence_map_part, u_s, v_s, 1);
			if (u_t != -1 && v_t != -1) {
				float transformed_d_t = *PointerAt<float>(
						depth_buffer_part, u_s, v_s);
				AddElementToCorrespondenceMap(correspondence_map,
						depth_buffer, u_s, v_s, u_t, v_t, transformed_d_t);
			}
		}
	}
}

int CountCorrespondence(const Image &correspondence_map)
{
	int correspondence_count = 0;
	for (int v_s = 0; v_s < correspondence_map.height_; v_s++) {
		for (int u_s = 0; u_s < correspondence_map.width_; u_s++) {
			int u_t = *PointerAt<int>(correspondence_map, u_s, v_s, 0);
			int v_t = *PointerAt<int>(correspondence_map, u_s, v_s, 1);
			if (u_t != -1 && v_t != -1) {
				correspondence_count++;
			}
		}
	}
	return correspondence_count;
}

std::shared_ptr<CorrespondenceSetPixelWise> ComputeCorrespondence(
		const Eigen::Matrix3d intrinsic_matrix,
		const Eigen::Matrix4d &extrinsic,
		const Image &depth_s, const Image &depth_t,
		const OdometryOption &option)
{
	const Eigen::Matrix3d K = intrinsic_matrix;
	const Eigen::Matrix3d K_inv = K.inverse();
	const Eigen::Matrix4d extrinsic_inv = extrinsic.inverse();
	const Eigen::Matrix3d R = extrinsic_inv.block<3, 3>(0, 0);
	const Eigen::Matrix3d KRK_inv = K * R * K_inv;
	Eigen::Vector3d Kt = K * extrinsic_inv.block<3, 1>(0, 3);

	std::shared_ptr<Image> correspondence_map;
	std::shared_ptr<Image> depth_buffer;
	std::tie(correspondence_map, depth_buffer) =
			InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);

#ifdef _OPENMP
#pragma omp parallel
	{
#endif
	std::shared_ptr<Image> correspondence_map_private;
	std::shared_ptr<Image> depth_buffer_private;
	std::tie(correspondence_map_private, depth_buffer_private) =
			InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);
#ifdef _OPENMP
#pragma omp for nowait
#endif
	for (int v_t = 0; v_t < depth_t.height_; v_t++) {
		for (int u_t = 0; u_t < depth_t.width_; u_t++) {
			double d_t = *PointerAt<float>(depth_t, u_t, v_t);
			if (!std::isnan(d_t)) {
				Eigen::Vector3d uv_in_t =
						d_t * KRK_inv * Eigen::Vector3d(u_t, v_t, 1.0) + Kt;
				double transformed_d_t = uv_in_t(2);
				int u_s = (int)(uv_in_t(0) / transformed_d_t + 0.5);
				int v_s = (int)(uv_in_t(1) / transformed_d_t + 0.5);
				if (u_s >= 0 && u_s < depth_t.width_ &&
					v_s >= 0 && v_s < depth_t.height_) {
					double d_s = *PointerAt<float>(depth_s, u_s, v_s);
					if (!std::isnan(d_s) && std::abs(transformed_d_t - d_s)
						<= option.max_depth_diff_) {
						AddElementToCorrespondenceMap(
								*correspondence_map_private,
								*depth_buffer_private,
								u_s, v_s, u_t, v_t, (float)transformed_d_t);
					}
				}
			}
		}
	}
#ifdef _OPENMP
#pragma omp critical
{
#endif
	MergeCorrespondenceMaps(
			*correspondence_map, *depth_buffer,
			*correspondence_map_private, *depth_buffer_private);
#ifdef _OPENMP
}	//	omp critical
}	//	omp parallel
#endif

	auto correspondence = std::make_shared<CorrespondenceSetPixelWise>();
	int correspondence_count = CountCorrespondence(*correspondence_map);
	correspondence->resize(correspondence_count);
	int cnt = 0;
	for (int v_s = 0; v_s < correspondence_map->height_; v_s++) {
		for (int u_s = 0; u_s < correspondence_map->width_; u_s++) {
			int u_t = *PointerAt<int>(*correspondence_map, u_s, v_s, 0);
			int v_t = *PointerAt<int>(*correspondence_map, u_s, v_s, 1);
			if (u_t != -1 && v_t != -1) {
				Eigen::Vector4i pixel_correspondence(u_s, v_s, u_t, v_t);
				(*correspondence)[cnt] = pixel_correspondence;
				cnt++;
			}
		}
	}
	return correspondence;
}

std::shared_ptr<Image> ConvertDepthImageToXYZImage(
		const Image &depth, const Eigen::Matrix3d &intrinsic_matrix)
{
	auto image_xyz = std::make_shared<Image>();
	if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
		PrintDebug("[ConvertDepthImageToXYZImage] Unsupported image format.\n");
		return image_xyz;
	}
	const double inv_fx = 1.0 / intrinsic_matrix(0, 0);
	const double inv_fy = 1.0 / intrinsic_matrix(1, 1);
	const double ox = intrinsic_matrix(0, 2);
	const double oy = intrinsic_matrix(1, 2);
	image_xyz->PrepareImage(depth.width_, depth.height_, 3, 4);

	for (int y = 0; y < image_xyz->height_; y++) {
		for (int x = 0; x < image_xyz->width_; x++) {
			float *px = PointerAt<float>(*image_xyz, x, y, 0);
			float *py = PointerAt<float>(*image_xyz, x, y, 1);
			float *pz = PointerAt<float>(*image_xyz, x, y, 2);
			float z = *PointerAt<float>(depth, x, y);
			*px = (float)((x - ox) * z * inv_fx);
			*py = (float)((y - oy) * z * inv_fy);
			*pz = z;
		}
	}
	return image_xyz;
}

std::vector<Eigen::Matrix3d>
		CreateCameraMatrixPyramid(
		const PinholeCameraIntrinsic &pinhole_camera_intrinsic, int levels)
{
	std::vector<Eigen::Matrix3d> pyramid_camera_matrix;
	pyramid_camera_matrix.reserve(levels);
	for (int i = 0; i < levels; i++) {
		Eigen::Matrix3d level_camera_matrix;
		if (i == 0)
			level_camera_matrix = pinhole_camera_intrinsic.intrinsic_matrix_;
		else
			level_camera_matrix = 0.5 * pyramid_camera_matrix[i - 1];
		level_camera_matrix(2, 2) = 1.;
		pyramid_camera_matrix.push_back(level_camera_matrix);
	}
	return pyramid_camera_matrix;
}

Eigen::Matrix6d CreateInfomationMatrix(
		const Eigen::Matrix4d &extrinsic,
		const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
		const Image &depth_s, const Image &depth_t,
		const OdometryOption &option)
{
	auto correspondence = ComputeCorrespondence(
			pinhole_camera_intrinsic.intrinsic_matrix_,
			extrinsic, depth_s, depth_t, option);

	auto xyz_t = ConvertDepthImageToXYZImage(
			depth_t, pinhole_camera_intrinsic.intrinsic_matrix_);

	// write q^*
	// see http://redwood-data.org/indoor/registration.html
	// note: I comes first and q_skew is scaled by factor 2.
	Eigen::Matrix6d GTG = Eigen::Matrix6d::Identity();
#ifdef _OPENMP
#pragma omp parallel
	{
#endif
		Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Identity();
		Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
		for (auto row = 0; row < correspondence->size(); row++) {
			int u_t = (*correspondence)[row](2);
			int v_t = (*correspondence)[row](3);
			double x = *PointerAt<float>(*xyz_t, u_t, v_t, 0);
			double y = *PointerAt<float>(*xyz_t, u_t, v_t, 1);
			double z = *PointerAt<float>(*xyz_t, u_t, v_t, 2);
			G_r_private.setZero();
			G_r_private(0) = 1.0;
			G_r_private(4) = 2.0 * z;
			G_r_private(5) = -2.0 * y;
			GTG_private.noalias() += G_r_private * G_r_private.transpose();
			G_r_private.setZero();
			G_r_private(1) = 1.0;
			G_r_private(3) = -2.0 * z;
			G_r_private(5) = 2.0 * x;
			GTG_private.noalias() += G_r_private * G_r_private.transpose();
			G_r_private.setZero();
			G_r_private(2) = 1.0;
			G_r_private(3) = 2.0 * y;
			G_r_private(4) = -2.0 * x;
			GTG_private.noalias() += G_r_private * G_r_private.transpose();
		}
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			GTG += GTG_private;
		}
#ifdef _OPENMP
	}
#endif
	return std::move(GTG);
}

void NormalizeIntensity(Image &image_s, Image &image_t,
		CorrespondenceSetPixelWise &correspondence)
{
	if (image_s.width_ != image_t.width_ ||
		image_s.height_ != image_t.height_) {
		PrintError("[NormalizeIntensity] Size of two input images should be same\n");
		return;
	}
	double mean_s = 0.0, mean_t = 0.0;
	for (int row = 0; row < correspondence.size(); row++) {
		int u_s = correspondence[row](0);
		int v_s = correspondence[row](1);
		int u_t = correspondence[row](2);
		int v_t = correspondence[row](3);
		mean_s += *PointerAt<float>(image_s, u_s, v_s);
		mean_t += *PointerAt<float>(image_t, u_t, v_t);
	}
	mean_s /= (double)correspondence.size();
	mean_t /= (double)correspondence.size();
	LinearTransformImage(image_s, 0.5 / mean_s, 0.0);
	LinearTransformImage(image_t, 0.5 / mean_t, 0.0);
}

inline std::shared_ptr<RGBDImage> PackRGBDImage(
	const Image &color, const Image &depth) {
	return std::make_shared<RGBDImage>(RGBDImage(color, depth));
}

std::shared_ptr<Image> PreprocessDepth(
		const Image &depth_orig, const OdometryOption &option)
{
	std::shared_ptr<Image> depth_processed = std::make_shared<Image>();
	*depth_processed = depth_orig;
	for (int y = 0; y < depth_processed->height_; y++) {
		for (int x = 0; x < depth_processed->width_; x++) {
			float *p = PointerAt<float>(*depth_processed, x, y);
			if ((*p < option.min_depth_ || *p > option.max_depth_ || *p <= 0))
				*p = std::numeric_limits<float>::quiet_NaN();
		}
	}
	return depth_processed;
}

inline bool CheckImagePair(const Image &image_s, const Image &image_t)
{
	return (image_s.width_ == image_t.width_ &&
			image_s.height_ == image_t.height_);
}

inline bool CheckRGBDImagePair(const RGBDImage &source, const RGBDImage &target)
{
	return (CheckImagePair(source.color_, target.color_) &&
			CheckImagePair(source.depth_, target.depth_) &&
			CheckImagePair(source.color_, source.depth_) &&
			CheckImagePair(target.color_, target.depth_) &&
			CheckImagePair(source.color_, target.color_) &&
			source.color_.num_of_channels_ == 1 &&
			source.depth_.num_of_channels_ == 1 &&
			target.color_.num_of_channels_ == 1 &&
			target.depth_.num_of_channels_ == 1 &&
			source.color_.bytes_per_channel_ == 4 &&
			target.color_.bytes_per_channel_ == 4 &&
			source.depth_.bytes_per_channel_ == 4 &&
			target.depth_.bytes_per_channel_ == 4);
}

std::tuple<std::shared_ptr<RGBDImage>, std::shared_ptr<RGBDImage>>
		InitializeRGBDOdometry(
		const RGBDImage &source, const RGBDImage &target,
		const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
		const Eigen::Matrix4d &odo_init,
		const OdometryOption &option)
{
	auto source_gray = FilterImage(source.color_,
			Image::FilterType::Gaussian3);
	auto target_gray = FilterImage(target.color_,
			Image::FilterType::Gaussian3);
	auto source_depth_preprocessed = PreprocessDepth(source.depth_, option);
	auto target_depth_preprocessed = PreprocessDepth(target.depth_, option);
	auto source_depth = FilterImage(*source_depth_preprocessed,
			Image::FilterType::Gaussian3);
	auto target_depth = FilterImage(*target_depth_preprocessed,
			Image::FilterType::Gaussian3);

	auto correspondence = ComputeCorrespondence(
			pinhole_camera_intrinsic.intrinsic_matrix_, odo_init,
			*source_depth, *target_depth, option);
	int corresps_count_required = (int)(source_gray->height_ *
			source_gray->width_ * option.minimum_correspondence_ratio_ + 0.5);
	if (correspondence->size() < corresps_count_required) {
		PrintWarning("[InitializeRGBDPair] Bad initial pose\n");
	}
	NormalizeIntensity(*source_gray, *target_gray, *correspondence);

	auto source_out = PackRGBDImage(*source_gray, *source_depth);
	auto target_out = PackRGBDImage(*target_gray, *target_depth);
	return std::make_tuple(source_out, target_out);
}

std::tuple<bool, Eigen::Matrix4d> DoSingleIteration(
	int iter, int level,
	const RGBDImage &source, const RGBDImage &target,
	const Image &source_xyz,
	const RGBDImage &target_dx, const RGBDImage &target_dy,
	const Eigen::Matrix3d intrinsic,
	const Eigen::Matrix4d &extrinsic_initial,
	const RGBDOdometryJacobian &jacobian_method,
	const OdometryOption &option)
{
	auto correspondence = ComputeCorrespondence(
			intrinsic, extrinsic_initial, source.depth_, target.depth_, option);
	int corresps_count_required = (int)(source.color_.height_ *
			source.color_.width_ * option.minimum_correspondence_ratio_ + 0.5);
	int corresps_count = (int)correspondence->size();
	if (corresps_count < corresps_count_required) {
		PrintWarning("[ComputeOdometry] Too fewer correspondences (%d found / %d required)\n",
				corresps_count, corresps_count_required);
		return std::make_tuple(false, Eigen::Matrix4d::Identity());
	}

	auto f_lambda = [&]
			(int i, std::vector<Eigen::Vector6d> &J_r, std::vector<double> &r) {
		jacobian_method.ComputeJacobianAndResidual(i, J_r, r,
				source, target, source_xyz, target_dx, target_dy,
				intrinsic, extrinsic_initial, *correspondence);
	};
	PrintDebug("Iter : %d, Level : %d, ", iter, level);
	Eigen::Matrix6d JTJ;
	Eigen::Vector6d JTr;
	std::tie(JTJ, JTr) = ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
			f_lambda, corresps_count);

	bool is_success;
	Eigen::Matrix4d extrinsic;
	std::tie(is_success, extrinsic) =
			SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
	if (!is_success) {
		PrintWarning("[ComputeOdometry] no solution!\n");
		return std::make_tuple(false, Eigen::Matrix4d::Identity());
	} else {
		return std::make_tuple(true, extrinsic);
	}
}

std::tuple<bool, Eigen::Matrix4d> ComputeMultiscale(
		const RGBDImage &source, const RGBDImage &target,
		const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
		const Eigen::Matrix4d &extrinsic_initial,
		const RGBDOdometryJacobian &jacobian_method,
		const OdometryOption &option)
{
	std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
	int num_levels = (int)iter_counts.size();

	auto source_pyramid = CreateRGBDImagePyramid(source, num_levels);
	auto target_pyramid = CreateRGBDImagePyramid(target, num_levels);
	auto target_pyramid_dx = FilterRGBDImagePyramid
			(target_pyramid, Image::FilterType::Sobel3Dx);
	auto target_pyramid_dy = FilterRGBDImagePyramid
			(target_pyramid, Image::FilterType::Sobel3Dy);

	Eigen::Matrix4d result_odo = extrinsic_initial.isZero() ?
			Eigen::Matrix4d::Identity() : extrinsic_initial;

	std::vector<Eigen::Matrix3d> pyramid_camera_matrix =
			CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
			(int)iter_counts.size());

	for (int level = num_levels - 1; level >= 0; level--) {
		const Eigen::Matrix3d level_camera_matrix = pyramid_camera_matrix[level];

		auto source_xyz_level = ConvertDepthImageToXYZImage(
				source_pyramid[level]->depth_, level_camera_matrix);
		auto source_level = PackRGBDImage(source_pyramid[level]->color_,
				source_pyramid[level]->depth_);
		auto target_level = PackRGBDImage(target_pyramid[level]->color_,
				target_pyramid[level]->depth_);
		auto target_dx_level = PackRGBDImage(target_pyramid_dx[level]->color_,
				target_pyramid_dx[level]->depth_);
		auto target_dy_level = PackRGBDImage(target_pyramid_dy[level]->color_,
				target_pyramid_dy[level]->depth_);

		for (int iter = 0; iter < iter_counts[num_levels - level - 1]; iter++) {
			Eigen::Matrix4d curr_odo;
			bool is_success;
			std::tie(is_success, curr_odo) = DoSingleIteration(
				iter, level,
				*source_level, *target_level, *source_xyz_level,
				*target_dx_level, *target_dy_level, level_camera_matrix,
				result_odo, jacobian_method, option);
			result_odo = curr_odo * result_odo;

			if (!is_success) {
				PrintWarning("[ComputeOdometry] no solution!\n");
				return std::make_tuple(false, Eigen::Matrix4d::Identity());
			}
		}
	}
	return std::make_tuple(true, result_odo);
}

}	// unnamed namespace

std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
		ComputeRGBDOdometry(const RGBDImage &source, const RGBDImage &target,
		const PinholeCameraIntrinsic &pinhole_camera_intrinsic
		/*= PinholeCameraIntrinsic()*/,
		const Eigen::Matrix4d &odo_init /*= Eigen::Matrix4d::Identity()*/,
		const RGBDOdometryJacobian &jacobian_method
		/*=RGBDOdometryJacobianFromHybridTerm*/,
		const OdometryOption &option /*= OdometryOption()*/)
{
	if (!CheckRGBDImagePair(source, target)) {
		PrintError("[RGBDOdometry] Two RGBD pairs should be same in size.\n");
		return std::make_tuple(false,
				Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Zero());
	}

	std::shared_ptr<RGBDImage> source_processed, target_processed;
	std::tie(source_processed, target_processed) =
			InitializeRGBDOdometry(source, target, pinhole_camera_intrinsic,
			odo_init, option);

	Eigen::Matrix4d extrinsic;
	bool is_success;
	std::tie(is_success, extrinsic) = ComputeMultiscale(
			*source_processed, *target_processed,
			pinhole_camera_intrinsic, odo_init, jacobian_method, option);

	if (is_success) {
		Eigen::Matrix4d trans_output = extrinsic;
		Eigen::MatrixXd info_output = CreateInfomationMatrix(extrinsic,
				pinhole_camera_intrinsic, source_processed->depth_,
				target_processed->depth_, option);
		return std::make_tuple(true, trans_output, info_output);
	}
	else {
		return std::make_tuple(false,
				Eigen::Matrix4d::Identity(), Eigen::Matrix6d::Identity());
	}
}

}	// namespace three
