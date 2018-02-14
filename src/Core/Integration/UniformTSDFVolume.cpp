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

#include "UniformTSDFVolume.h"

#include <unordered_map>
#include <thread>

#include <Core/Utility/Helper.h>
#include <Core/Integration/MarchingCubesConst.h>

namespace three{

UniformTSDFVolume::UniformTSDFVolume(double length, int resolution,
		double sdf_trunc, bool with_color,
		const Eigen::Vector3d &origin/* = Eigen::Vector3d::Zero()*/) :
		TSDFVolume(length / (double)resolution, sdf_trunc, with_color),
		origin_(origin), length_(length), resolution_(resolution),
		voxel_num_(resolution * resolution * resolution),
		tsdf_(voxel_num_, 0.0f), color_(with_color ? voxel_num_ : 0,
		Eigen::Vector3f::Zero()), weight_(voxel_num_, 0.0f)
{
}

UniformTSDFVolume::~UniformTSDFVolume()
{
}

void UniformTSDFVolume::Reset()
{
	std::memset(tsdf_.data(), 0, voxel_num_ * 4);
	std::memset(weight_.data(), 0, voxel_num_ * 4);
	if (with_color_) {
		std::memset(color_.data(), 0, voxel_num_ * 12);
	}
}

void UniformTSDFVolume::Integrate(const RGBDImage &image,
		const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic)
{
	// This function goes through the voxels, and scan convert the relative
	// depth/color value into the voxel.
	// The following implementation is a highly optimized version.
	if ((image.depth_.num_of_channels_ != 1) ||
			(image.depth_.bytes_per_channel_ != 4) ||
			(image.depth_.width_ != intrinsic.width_) ||
			(image.depth_.height_ != intrinsic.height_) ||
			(with_color_ && image.color_.num_of_channels_ != 3) ||
			(with_color_ && image.color_.bytes_per_channel_ != 1) ||
			(with_color_ && image.color_.width_ != intrinsic.width_) ||
			(with_color_ && image.color_.height_ != intrinsic.height_)) {
		PrintWarning("[UniformTSDFVolume::Integrate] Unsupported image format. Please check if you have called CreateRGBDImageFromColorAndDepth() with convert_rgb_to_intensity=false.\n");
		return;
	}
	auto depth2cameradistance = CreateDepthToCameraDistanceMultiplierFloatImage(
			intrinsic);
	IntegrateWithDepthToCameraDistanceMultiplier(image, intrinsic,
			extrinsic, *depth2cameradistance);
}

std::shared_ptr<PointCloud> UniformTSDFVolume::ExtractPointCloud()
{
	auto pointcloud = std::make_shared<PointCloud>();
	double half_voxel_length = voxel_length_ * 0.5;
	for (int x = 1; x < resolution_ - 1; x++) {
		for (int y = 1; y < resolution_ - 1; y++) {
			for (int z = 1; z < resolution_ - 1; z++) {
				Eigen::Vector3i idx0(x, y, z);
				float w0 = weight_[IndexOf(idx0)];
				float f0 = tsdf_[IndexOf(idx0)];
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
							float w1 = weight_[IndexOf(idx1)];
							float f1 = tsdf_[IndexOf(idx1)];
							if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
									f0 * f1 < 0) {
								float r0 = std::fabs(f0);
								float r1 = std::fabs(f1);
								Eigen::Vector3d p = p0;
								p(i) = (p0(i) * r1 + p1(i) * r0) / (r0 + r1);
								pointcloud->points_.push_back(p + origin_);
								if (with_color_) {
									pointcloud->colors_.push_back(
											((color_[IndexOf(idx0)] * r1 +
											color_[IndexOf(idx1)] * r0) /
											(r0 + r1) / 255.0f).cast<double>());
								}
								// has_normal
								pointcloud->normals_.push_back(GetNormalAt(p));
							}
						}
					}
				}
			}
		}
	}
	return pointcloud;
}

std::shared_ptr<TriangleMesh> UniformTSDFVolume::ExtractTriangleMesh()
{
	// implementation of marching cubes, based on
	// http://paulbourke.net/geometry/polygonise/
	auto mesh = std::make_shared<TriangleMesh>();
	double half_voxel_length = voxel_length_ * 0.5;
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
					if (weight_[IndexOf(idx)] == 0.0f) {
						cube_index = 0;
						break;
					} else {
						f[i] = tsdf_[IndexOf(idx)];
						if (f[i] < 0.0f) {
							cube_index |= (1 << i);
						}
						if (with_color_) {
							c[i] = color_[IndexOf(idx)].cast<double>() / 255.0;
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
							edge_to_index[i] = (int)mesh->vertices_.size();
							edgeindex_to_vertexindex[edge_index] =
									(int)mesh->vertices_.size();
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
							mesh->vertices_.push_back(pt + origin_);
							if (with_color_) {
								const auto &c0 = c[edge_to_vert[i][0]];
								const auto &c1 = c[edge_to_vert[i][1]];
								mesh->vertex_colors_.push_back(
										(f1 * c0 + f0 * c1) / (f0 + f1));
							}
						} else {
							edge_to_index[i] = edgeindex_to_vertexindex.find(
									edge_index)->second;
						}
					}
				}
				for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
					mesh->triangles_.push_back(Eigen::Vector3i(
							edge_to_index[tri_table[cube_index][i]],
							edge_to_index[tri_table[cube_index][i + 2]],
							edge_to_index[tri_table[cube_index][i + 1]]));
				}
			}
		}
	}
	return mesh;
}

std::shared_ptr<PointCloud> UniformTSDFVolume::ExtractVoxelPointCloud()
{
	auto voxel = std::make_shared<PointCloud>();
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
					voxel->points_.push_back(pt + origin_);
					double c = (static_cast<double>(*p_tsdf) + 1.0) * 0.5;
					voxel->colors_.push_back(Eigen::Vector3d(c, c, c));
				}
			}
		}
	}
	return voxel;
}

void UniformTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier(
		const RGBDImage &image, const PinholeCameraIntrinsic &intrinsic,
		const Eigen::Matrix4d &extrinsic,
		const Image &depth_to_camera_distance_multiplier)
{
	const float fx = static_cast<float>(intrinsic.GetFocalLength().first);
	const float fy = static_cast<float>(intrinsic.GetFocalLength().second);
	const float cx = static_cast<float>(intrinsic.GetPrincipalPoint().first);
	const float cy = static_cast<float>(intrinsic.GetPrincipalPoint().second);
	const Eigen::Matrix4f extrinsic_f = extrinsic.cast<float>();
	const float voxel_length_f = static_cast<float>(voxel_length_);
	const float half_voxel_length_f = voxel_length_f * 0.5f;
	const float sdf_trunc_f = static_cast<float>(sdf_trunc_);
	const float sdf_trunc_inv_f = 1.0f / sdf_trunc_f;
	const Eigen::Matrix4f extrinsic_scaled_f = extrinsic_f *
			voxel_length_f;
	const float safe_width_f = intrinsic.width_ - 0.0001f;
	const float safe_height_f = intrinsic.height_ - 0.0001f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int x = 0; x < resolution_; x++) {
		for (int y = 0; y < resolution_; y++) {
			int idx_shift = x * resolution_ * resolution_ + y * resolution_;
			float *p_tsdf = (float *)tsdf_.data() + idx_shift;
			float *p_weight = (float *)weight_.data() + idx_shift;
			float *p_color = (float *)color_.data() + idx_shift * 3;
			Eigen::Vector4f voxel_pt_camera = extrinsic_f * Eigen::Vector4f(
					half_voxel_length_f + voxel_length_f * x +
					(float)origin_(0),
					half_voxel_length_f + voxel_length_f * y +
					(float)origin_(1),
					half_voxel_length_f + (float)origin_(2),
					1.0f);
			for (int z = 0; z < resolution_; z++,
					voxel_pt_camera(0) += extrinsic_scaled_f(0, 2),
					voxel_pt_camera(1) += extrinsic_scaled_f(1, 2),
					voxel_pt_camera(2) += extrinsic_scaled_f(2, 2),
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
						float d = *PointerAt<float>(image.depth_, u, v);
						if (d > 0.0f) {
							float sdf = (d - voxel_pt_camera(2)) * (
									*PointerAt<float>(
									depth_to_camera_distance_multiplier,
									u, v));
							if (sdf > -sdf_trunc_f) {
								// integrate
								float tsdf = std::min(1.0f,
										sdf * sdf_trunc_inv_f);
								*p_tsdf = ((*p_tsdf) * (*p_weight) + tsdf) /
										(*p_weight + 1.0f);
								if (with_color_) {
									const uint8_t *rgb = PointerAt<uint8_t>(
											image.color_, u, v, 0);
									p_color[0] = (p_color[0] *
											(*p_weight) + rgb[0]) /
											(*p_weight + 1.0f);
									p_color[1] = (p_color[1] *
											(*p_weight) + rgb[1]) /
											(*p_weight + 1.0f);
									p_color[2] = (p_color[2] *
											(*p_weight) + rgb[2]) /
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

Eigen::Vector3d UniformTSDFVolume::GetNormalAt(const Eigen::Vector3d &p)
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

double UniformTSDFVolume::GetTSDFAt(const Eigen::Vector3d &p)
{
	Eigen::Vector3i idx;
	Eigen::Vector3d p_grid = p / voxel_length_ - Eigen::Vector3d(0.5, 0.5, 0.5);
	for (int i = 0; i < 3; i++) {
		idx(i) = (int)std::floor(p_grid(i));
	}
	Eigen::Vector3d r = p_grid - idx.cast<double>();
	return (1 - r(0)) * (
			(1 - r(1)) * (
			(1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 0, 0))] +
			r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 0, 1))]
			) + r(1) * (
			(1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 1, 0))] +
			r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(0, 1, 1))]
			)) + r(0) * (
			(1 - r(1)) * (
			(1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 0, 0))] +
			r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 0, 1))]
			) + r(1) * (
			(1 - r(2)) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 1, 0))] +
			r(2) * tsdf_[IndexOf(idx + Eigen::Vector3i(1, 1, 1))]
			));
}

}	// namespace three
