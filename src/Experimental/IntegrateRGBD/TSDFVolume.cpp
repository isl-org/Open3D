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

#include "TSDFVolume.h"

#include <unordered_map>
#include <thread>

#include <Eigen/Dense>
#include <Core/Utility/Helper.h>
#include "Helper.h"

namespace three{

TSDFVolume::TSDFVolume(double length, int resolution, double sdf_trunc,
		bool has_color) : length_(length), resolution_(resolution),
		voxel_length_(length / (double)resolution),
		voxel_num_(resolution_ * resolution_ * resolution_),
		sdf_trunc_(sdf_trunc), has_color_(has_color), tsdf_(voxel_num_),
		color_(has_color ? voxel_num_ * 3 : 0), weight_(voxel_num_)
{
}

TSDFVolume::~TSDFVolume()
{
}

void TSDFVolume::Reset()
{
	std::memset(tsdf_.data(), 0, resolution_ * resolution_ * resolution_ * 4);
	std::memset(weight_.data(), 0, resolution_ * resolution_ * resolution_ * 4);
	if (has_color_) {
		std::memset(color_.data(), 0,
				resolution_ * resolution_ * resolution_ * 12);
	}
}

/*
void TSDFVolume::Integrate(const Image &depth_f, const Image &color,
		const Image &depth2cameradistance,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
	// This function goes through the voxels, and scan convert the relative
	// depth/color value into the voxel.
	// The following implementation is a highly optimized version.
	const float fx = static_cast<float>(intrinsic.GetFocalLength().first);
	const float fy = static_cast<float>(intrinsic.GetFocalLength().second);
	const float cx = static_cast<float>(intrinsic.GetPrincipalPoint().first);
	const float cy = static_cast<float>(intrinsic.GetPrincipalPoint().second);
	const Eigen::Matrix4f extrinsic_inv_f = extrinsic.inverse().cast<float>();
	const float voxel_length_f = static_cast<float>(voxel_length_);
	const float half_voxel_length_f = voxel_length_f * 0.5f;
	const float sdf_trunc_f = static_cast<float>(sdf_trunc_);
	const float sdf_trunc_inv_f = 1.0f / sdf_trunc_f;
	const Eigen::Matrix4f extrinsic_inv_scaled_f = extrinsic_inv_f *
			voxel_length_f;
	const float safe_width_f = intrinsic.width_ - 0.0001f;
	const float safe_height_f = intrinsic.height_ - 0.0001f;
	
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
	for (int x = 0; x < resolution_; x++) {
		for (int y = 0; y < resolution_; y++) {
			int idx_shift = x * resolution_ * resolution_ + y * resolution_;
			float *p_tsdf = (float *)tsdf_.data() + idx_shift;
			float *p_weight = (float *)weight_.data() + idx_shift;
			float *p_color = (float *)color_.data() + idx_shift * 3;
			Eigen::Vector4f voxel_pt_camera = extrinsic_inv_f * Eigen::Vector4f(
					half_voxel_length_f + voxel_length_f * x,
					half_voxel_length_f + voxel_length_f * y,
					half_voxel_length_f,
					1.0f);
			for (int z = 0; z < resolution_; z++,
					voxel_pt_camera(0) += extrinsic_inv_scaled_f(0, 2),
					voxel_pt_camera(1) += extrinsic_inv_scaled_f(1, 2),
					voxel_pt_camera(2) += extrinsic_inv_scaled_f(2, 2),
					p_tsdf++, p_weight++, p_color += 3) {
				if (voxel_pt_camera(2) > 0) {
					float u_f = voxel_pt_camera(0) * fx /
							voxel_pt_camera(2) + cx + 0.5f;
					float v_f = voxel_pt_camera(1) * fy /
							voxel_pt_camera(2) + cy + 0.5f;
					if (u_f >= 0.0001f && u_f < safe_width_f &&
							v_f >= 0.0001f && v_f < safe_height_f) {
						int u = (int)u_f;
						int v = (int)v_f;
						float d = *PointerAt<float>(depth_f, u, v);
						if (d > 0.0f) {
							float sdf = (d - voxel_pt_camera(2)) * (
									*PointerAt<float>(depth2cameradistance, u,
									v));
							if (sdf > -sdf_trunc_f) {
								// integrate
								float tsdf = std::min(1.0f,
										sdf * sdf_trunc_inv_f);
								*p_tsdf = ((*p_tsdf) * (*p_weight) + tsdf) /
										(*p_weight + 1.0f);
								if (has_color_) {
									const RGB *rgb = PointerAt<RGB>(color, u,
											v);
									p_color[0] = (p_color[0] *
											(*p_weight) + rgb->rgb[0]) /
											(*p_weight + 1.0f);
									p_color[1] = (p_color[1] *
											(*p_weight) + rgb->rgb[1]) /
											(*p_weight + 1.0f);
									p_color[2] = (p_color[2] *
											(*p_weight) + rgb->rgb[2]) /
											(*p_weight + 1.0f);
								}
								*p_weight += 1.0f;
							}
						}
					}
				}
			}
		}
	}
}
*/

void TSDFVolume::Integrate_thread(const Image &depth_f, const Image &color,
		const Image &depth2cameradistance,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic, int x_begin, int x_end)
{
	// This function goes through the voxels, and scan convert the relative
	// depth/color value into the voxel.
	// The following implementation is a highly optimized version.
	const float fx = static_cast<float>(intrinsic.GetFocalLength().first);
	const float fy = static_cast<float>(intrinsic.GetFocalLength().second);
	const float cx = static_cast<float>(intrinsic.GetPrincipalPoint().first);
	const float cy = static_cast<float>(intrinsic.GetPrincipalPoint().second);
	const Eigen::Matrix4f extrinsic_inv_f = extrinsic.inverse().cast<float>();
	const float voxel_length_f = static_cast<float>(voxel_length_);
	const float half_voxel_length_f = voxel_length_f * 0.5f;
	const float sdf_trunc_f = static_cast<float>(sdf_trunc_);
	const float sdf_trunc_inv_f = 1.0f / sdf_trunc_f;
	const Eigen::Matrix4f extrinsic_inv_scaled_f = extrinsic_inv_f *
			voxel_length_f;
	const float safe_width_f = intrinsic.width_ - 0.0001f;
	const float safe_height_f = intrinsic.height_ - 0.0001f;
	
	for (int x = x_begin; x < x_end; x++) {
		for (int y = 0; y < resolution_; y++) {
			int idx_shift = x * resolution_ * resolution_ + y * resolution_;
			float *p_tsdf = (float *)tsdf_.data() + idx_shift;
			float *p_weight = (float *)weight_.data() + idx_shift;
			float *p_color = (float *)color_.data() + idx_shift * 3;
			Eigen::Vector4f voxel_pt_camera = extrinsic_inv_f * Eigen::Vector4f(
					half_voxel_length_f + voxel_length_f * x,
					half_voxel_length_f + voxel_length_f * y,
					half_voxel_length_f,
					1.0f);
			for (int z = 0; z < resolution_; z++,
					voxel_pt_camera(0) += extrinsic_inv_scaled_f(0, 2),
					voxel_pt_camera(1) += extrinsic_inv_scaled_f(1, 2),
					voxel_pt_camera(2) += extrinsic_inv_scaled_f(2, 2),
					p_tsdf++, p_weight++, p_color += 3) {
				if (voxel_pt_camera(2) > 0) {
					float u_f = voxel_pt_camera(0) * fx /
							voxel_pt_camera(2) + cx + 0.5f;
					float v_f = voxel_pt_camera(1) * fy /
							voxel_pt_camera(2) + cy + 0.5f;
					if (u_f >= 0.0001f && u_f < safe_width_f &&
							v_f >= 0.0001f && v_f < safe_height_f) {
						int u = (int)u_f;
						int v = (int)v_f;
						float d = *PointerAt<float>(depth_f, u, v);
						if (d > 0.0f) {
							float sdf = (d - voxel_pt_camera(2)) * (
									*PointerAt<float>(depth2cameradistance, u,
									v));
							if (sdf > -sdf_trunc_f) {
								// integrate
								float tsdf = std::min(1.0f,
										sdf * sdf_trunc_inv_f);
								*p_tsdf = ((*p_tsdf) * (*p_weight) + tsdf) /
										(*p_weight + 1.0f);
								if (has_color_) {
									const RGB *rgb = PointerAt<RGB>(color, u,
											v);
									p_color[0] = (p_color[0] *
											(*p_weight) + rgb->rgb[0]) /
											(*p_weight + 1.0f);
									p_color[1] = (p_color[1] *
											(*p_weight) + rgb->rgb[1]) /
											(*p_weight + 1.0f);
									p_color[2] = (p_color[2] *
											(*p_weight) + rgb->rgb[2]) /
											(*p_weight + 1.0f);
								}
								*p_weight += 1.0f;
							}
						}
					}
				}
			}
		}
	}
}

void TSDFVolume::Integrate(const Image &depth_f, const Image &color,
		const Image &depth2cameradistance,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
	const int num_threads = 16;
	const int res = resolution_ / num_threads;
	std::vector<std::thread> thread_pool;
	for (int i = 0; i < num_threads; i++) {
		thread_pool.push_back(std::thread(&TSDFVolume::Integrate_thread, this,
			depth_f, color, depth2cameradistance, intrinsic, extrinsic,
			i * res, (i+1) * res));
	}
	for (int i = 0; i < num_threads; i++) {
		thread_pool[i].join();
	}
}

void TSDFVolume::ExtractVoxelPointCloud(PointCloud &voxel)
{
	voxel.Clear();
	double half_voxel_length = voxel_length_ * 0.5;
	float *p_tsdf = (float *)tsdf_.data();
	float *p_weight = (float *)weight_.data();
	float *p_color = (float *)color_.data();
	for (int x = 0; x < resolution_; x++) {
		for (int y = 0; y < resolution_; y++) {
			Eigen::Vector3d pt(
					half_voxel_length + voxel_length_ * x,
					half_voxel_length + voxel_length_ * y,
					half_voxel_length);
			for (int z = 0; z < resolution_; z++, pt(2) += voxel_length_,
					p_tsdf++, p_weight++, p_color += 3) {
				if (*p_weight != 0.0f && *p_tsdf < 0.98f &&
						*p_tsdf >= -0.98f ) {
					voxel.points_.push_back(pt);
					double c = (static_cast<double>(*p_tsdf) + 1.0) * 0.5;
					voxel.colors_.push_back(Eigen::Vector3d(c, c, c));
				}
			}
		}
	}
}

void TSDFVolume::ExtractPointCloud(PointCloud &pointcloud)
{
	pointcloud.Clear();
	double half_voxel_length = voxel_length_ * 0.5;
	for (int x = 1; x < resolution_ - 1; x++) {
		for (int y = 1; y < resolution_ - 1; y++) {
			for (int z = 1; z < resolution_ - 1; z++) {
				Eigen::Vector3i idx0(x, y, z);
				float w0 = weight_[index(idx0)];
				float f0 = tsdf_[index(idx0)];
				if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f) {
					Eigen::Vector3d p0(
							half_voxel_length + voxel_length_ * x,
							half_voxel_length + voxel_length_ * y,
							half_voxel_length + voxel_length_ * z);
					for (int i = 0; i < 3; i++) {
						Eigen::Vector3d p1 = p0;
						p1(i) += voxel_length_;
						Eigen::Vector3i idx1 = idx0;
						idx1(i) += 1;
						if (idx1(i) < resolution_ - 1) {
							float w1 = weight_[index(idx1)];
							float f1 = tsdf_[index(idx1)];
							if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
									f0 * f1 < 0) {
								float r0 = std::fabs(f0);
								float r1 = std::fabs(f1);
								Eigen::Vector3d p = p0;
								p(i) = (p0(i) * r1 + p1(i) * r0) / (r0 + r1);
								pointcloud.points_.push_back(p);
								if (has_color_) {
									pointcloud.colors_.push_back(
											((color_[index(idx0)] * r1 +
											color_[index(idx1)] * r0) /
											(r0 + r1) / 255.0f).cast<double>());
								}
								// has_normal
								pointcloud.normals_.push_back(GetNormalAt(p));
							}
						}
					}
				}
			}
		}
	}
}

namespace {

const int edge_table[256]={
		0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

const int tri_table[256][16] =
		{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
}  // unnamed namespace

void TSDFVolume::ExtractTriangleMesh(TriangleMesh &mesh)
{
	// implementation of marching cubes, based on
	// http://paulbourke.net/geometry/polygonise/
	const Eigen::Vector3i shift[8] = {
		Eigen::Vector3i(0, 0, 0),
		Eigen::Vector3i(1, 0, 0),
		Eigen::Vector3i(1, 1, 0),
		Eigen::Vector3i(0, 1, 0),
		Eigen::Vector3i(0, 0, 1),
		Eigen::Vector3i(1, 0, 1),
		Eigen::Vector3i(1, 1, 1),
		Eigen::Vector3i(0, 1, 1),
	};
	
	const Eigen::Vector4i edge_shift[12] = {
		Eigen::Vector4i(0, 0, 0, 0),
		Eigen::Vector4i(1, 0, 0, 1),
		Eigen::Vector4i(0, 1, 0, 0),
		Eigen::Vector4i(0, 0, 0, 1),
		Eigen::Vector4i(0, 0, 1, 0),
		Eigen::Vector4i(1, 0, 1, 1),
		Eigen::Vector4i(0, 1, 1, 0),
		Eigen::Vector4i(0, 0, 1, 1),
		Eigen::Vector4i(0, 0, 0, 2),
		Eigen::Vector4i(1, 0, 0, 2),
		Eigen::Vector4i(1, 1, 0, 2),
		Eigen::Vector4i(0, 1, 0, 2),
	};
	
	const int edge_to_vert[12][2] = {
		{0, 1},
		{1, 2},
		{3, 2},
		{0, 3},
		{4, 5},
		{5, 6},
		{7, 6},
		{4, 7},
		{0, 4},
		{1, 5},
		{2, 6},
		{3, 7},
	};
	
	double half_voxel_length = voxel_length_ * 0.5;
	mesh.Clear();
	std::unordered_map<Eigen::Vector4i, int, hash_eigen::hash<Eigen::Vector4i>>
			edgeindex_to_vertexindex;
	int edge_to_index[12];
	for (int x = 0; x < resolution_ - 1; x++) {
		for (int y = 0; y < resolution_ - 1; y++) {
			for (int z = 0; z < resolution_ - 1; z++) {
				int cube_index = 0;
				float f[8];
				Eigen::Vector3d c[8];
				for (int i = 0; i < 8; i++ ) {
					Eigen::Vector3i idx = Eigen::Vector3i(x, y, z) + shift[i];
					if (weight_[index(idx)] == 0.0) {
						cube_index = 0;
						break;
					} else {
						f[i] = tsdf_[index(idx)];
						if (f[i] < 0.0f) {
							cube_index |= (1 << i);
						}
						if (has_color_) {
							c[i] = color_[index(idx)].cast<double>() / 255.0;
						}
					}
				}
				if (cube_index == 0 || cube_index == 255) {
					continue;
				}
				for (int i = 0; i < 12; i++) {
					if (edge_table[cube_index] & (1 << i)) {
						Eigen::Vector4i edge_index =
								Eigen::Vector4i(x, y, z, 0) + edge_shift[i];
						if (edgeindex_to_vertexindex.find(edge_index) ==
								edgeindex_to_vertexindex.end()) {
							edge_to_index[i] = (int)mesh.vertices_.size();
							edgeindex_to_vertexindex[edge_index] =
									(int)mesh.vertices_.size();
							Eigen::Vector3d pt(
									half_voxel_length +
									voxel_length_ * edge_index(0),
									half_voxel_length +
									voxel_length_ * edge_index(1),
									half_voxel_length +
									voxel_length_ * edge_index(2));
							double f0 = std::abs((double)f[edge_to_vert[i][0]]);
							double f1 = std::abs((double)f[edge_to_vert[i][1]]);
							pt(edge_index(3)) += f0 * voxel_length_ / (f0 + f1);
							mesh.vertices_.push_back(pt);
							if (has_color_) {
								const auto &c0 = c[edge_to_vert[i][0]];
								const auto &c1 = c[edge_to_vert[i][1]];
								mesh.vertex_colors_.push_back(
										(f1 * c0 + f0 * c1) / (f0 + f1));
							}
						} else {
							edge_to_index[i] = edgeindex_to_vertexindex.find(
									edge_index)->second;
						}
					}
				}
				for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
					mesh.triangles_.push_back(Eigen::Vector3i(
							edge_to_index[tri_table[cube_index][i]],
							edge_to_index[tri_table[cube_index][i + 2]],
							edge_to_index[tri_table[cube_index][i + 1]]));
				}
			}
		}
	}
}

Eigen::Vector3d TSDFVolume::GetNormalAt(const Eigen::Vector3d &p)
{
	Eigen::Vector3d n;
	const double half_gap = 0.99 * voxel_length_;
	for (int i = 0; i < 3; i++) {
		Eigen::Vector3d p0 = p;
		p0(i) -= half_gap;
		Eigen::Vector3d p1 = p;
		p1(i) += half_gap;
		n(i) = GetTSDFAt(p1) - GetTSDFAt(p0);
	}
	return n.normalized();
}

double TSDFVolume::GetTSDFAt(const Eigen::Vector3d &p)
{
	Eigen::Vector3i idx;
	Eigen::Vector3d p_grid = p / voxel_length_ - Eigen::Vector3d(0.5, 0.5, 0.5);
	for (int i = 0; i < 3; i++) {
		idx(i) = (int)std::floor(p_grid(i));
	}
	Eigen::Vector3d r = p_grid - idx.cast<double>();
	return (1 - r(0)) * (
			(1 - r(1)) * (
			(1 - r(2)) * tsdf_[index(idx + Eigen::Vector3i(0, 0, 0))] +
			r(2) * tsdf_[index(idx + Eigen::Vector3i(0, 0, 1))]
			) + r(1) * (
			(1 - r(2)) * tsdf_[index(idx + Eigen::Vector3i(0, 1, 0))] +
			r(2) * tsdf_[index(idx + Eigen::Vector3i(0, 1, 1))]
			)) + r(0) * (
			(1 - r(1)) * (
			(1 - r(2)) * tsdf_[index(idx + Eigen::Vector3i(1, 0, 0))] +
			r(2) * tsdf_[index(idx + Eigen::Vector3i(1, 0, 1))]
			) + r(1) * (
			(1 - r(2)) * tsdf_[index(idx + Eigen::Vector3i(1, 1, 0))] +
			r(2) * tsdf_[index(idx + Eigen::Vector3i(1, 1, 1))]
			));
}

}	// namespace three
