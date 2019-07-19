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

#include "Open3D/Integration/UniformTSDFVolume.h"

#include <iostream>
#include <thread>
#include <unordered_map>

#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Integration/MarchingCubesConst.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace integration {

UniformTSDFVolume::UniformTSDFVolume(
        double length,
        int resolution,
        double sdf_trunc,
        TSDFVolumeColorType color_type,
        const Eigen::Vector3d &origin /* = Eigen::Vector3d::Zero()*/)
    : TSDFVolume(length / (double)resolution, sdf_trunc, color_type),
      origin_(origin),
      length_(length),
      resolution_(resolution),
      voxel_num_(resolution * resolution * resolution) {
    voxels_.resize(voxel_num_);
}

UniformTSDFVolume::~UniformTSDFVolume() {}

void UniformTSDFVolume::Reset() { voxels_.clear(); }

void UniformTSDFVolume::Integrate(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic) {
    // This function goes through the voxels, and scan convert the relative
    // depth/color value into the voxel.
    // The following implementation is a highly optimized version.
    if ((image.depth_.num_of_channels_ != 1) ||
        (image.depth_.bytes_per_channel_ != 4) ||
        (image.depth_.width_ != intrinsic.width_) ||
        (image.depth_.height_ != intrinsic.height_) ||
        (color_type_ == TSDFVolumeColorType::RGB8 &&
         image.color_.num_of_channels_ != 3) ||
        (color_type_ == TSDFVolumeColorType::RGB8 &&
         image.color_.bytes_per_channel_ != 1) ||
        (color_type_ == TSDFVolumeColorType::Gray32 &&
         image.color_.num_of_channels_ != 1) ||
        (color_type_ == TSDFVolumeColorType::Gray32 &&
         image.color_.bytes_per_channel_ != 4) ||
        (color_type_ != TSDFVolumeColorType::None &&
         image.color_.width_ != intrinsic.width_) ||
        (color_type_ != TSDFVolumeColorType::None &&
         image.color_.height_ != intrinsic.height_)) {
        utility::LogWarning(
                "[UniformTSDFVolume::Integrate] Unsupported image format.\n");
        return;
    }
    auto depth2cameradistance =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);
    IntegrateWithDepthToCameraDistanceMultiplier(image, intrinsic, extrinsic,
                                                 *depth2cameradistance);
}

std::shared_ptr<geometry::PointCloud> UniformTSDFVolume::ExtractPointCloud() {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    double half_voxel_length = voxel_length_ * 0.5;
    for (int x = 1; x < resolution_ - 1; x++) {
        for (int y = 1; y < resolution_ - 1; y++) {
            for (int z = 1; z < resolution_ - 1; z++) {
                Eigen::Vector3i idx0(x, y, z);
                float w0 = voxels_[IndexOf(idx0)].weight_;
                float f0 = voxels_[IndexOf(idx0)].tsdf_;
                const Eigen::Vector3d &c0 = voxels_[IndexOf(idx0)].color_;

                if (!(w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f)) {
                    continue;
                }
                Eigen::Vector3d p0(half_voxel_length + voxel_length_ * x,
                                   half_voxel_length + voxel_length_ * y,
                                   half_voxel_length + voxel_length_ * z);
                for (int i = 0; i < 3; i++) {
                    Eigen::Vector3d p1 = p0;
                    p1(i) += voxel_length_;
                    Eigen::Vector3i idx1 = idx0;
                    idx1(i) += 1;
                    if (idx1(i) < resolution_ - 1) {
                        float w1 = voxels_[IndexOf(idx1)].weight_;
                        float f1 = voxels_[IndexOf(idx1)].tsdf_;
                        const Eigen::Vector3d &c1 =
                                voxels_[IndexOf(idx1)].color_;
                        if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
                            f0 * f1 < 0) {
                            float r0 = std::fabs(f0);
                            float r1 = std::fabs(f1);
                            Eigen::Vector3d p = p0;
                            p(i) = (p0(i) * r1 + p1(i) * r0) / (r0 + r1);
                            pointcloud->points_.push_back(p + origin_);
                            if (color_type_ == TSDFVolumeColorType::RGB8) {
                                pointcloud->colors_.push_back(
                                        ((c0 * r1 + c1 * r0) / (r0 + r1) /
                                         255.0f)
                                                .cast<double>());
                            } else if (color_type_ ==
                                       TSDFVolumeColorType::Gray32) {
                                pointcloud->colors_.push_back(
                                        ((c0 * r1 + c1 * r0) / (r0 + r1))
                                                .cast<double>());
                            }
                            // has_normal
                            pointcloud->normals_.push_back(GetNormalAt(p));
                        }
                    }
                }
            }
        }
    }
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh>
UniformTSDFVolume::ExtractTriangleMesh() {
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    double half_voxel_length = voxel_length_ * 0.5;
    // Map of "edge_index = (x, y, z, 0) + edge_shift" to "global vertex index"
    std::unordered_map<
            Eigen::Vector4i, int, utility::hash_eigen::hash<Eigen::Vector4i>,
            std::equal_to<Eigen::Vector4i>,
            Eigen::aligned_allocator<std::pair<const Eigen::Vector4i, int>>>
            edgeindex_to_vertexindex;
    int edge_to_index[12];
    for (int x = 0; x < resolution_ - 1; x++) {
        for (int y = 0; y < resolution_ - 1; y++) {
            for (int z = 0; z < resolution_ - 1; z++) {
                int cube_index = 0;
                float f[8];
                Eigen::Vector3d c[8];
                for (int i = 0; i < 8; i++) {
                    Eigen::Vector3i idx = Eigen::Vector3i(x, y, z) + shift[i];

                    if (voxels_[IndexOf(idx)].weight_ == 0.0f) {
                        cube_index = 0;
                        break;
                    } else {
                        f[i] = voxels_[IndexOf(idx)].tsdf_;
                        if (f[i] < 0.0f) {
                            cube_index |= (1 << i);
                        }
                        if (color_type_ == TSDFVolumeColorType::RGB8) {
                            c[i] = voxels_[IndexOf(idx)].color_.cast<double>() /
                                   255.0;
                        } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                            c[i] = voxels_[IndexOf(idx)].color_.cast<double>();
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
                            if (color_type_ != TSDFVolumeColorType::None) {
                                const auto &c0 = c[edge_to_vert[i][0]];
                                const auto &c1 = c[edge_to_vert[i][1]];
                                mesh->vertex_colors_.push_back(
                                        (f1 * c0 + f0 * c1) / (f0 + f1));
                            }
                        } else {
                            edge_to_index[i] =
                                    edgeindex_to_vertexindex.find(edge_index)
                                            ->second;
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

std::shared_ptr<geometry::PointCloud>
UniformTSDFVolume::ExtractVoxelPointCloud() const {
    auto voxel = std::make_shared<geometry::PointCloud>();
    double half_voxel_length = voxel_length_ * 0.5;
    // const float *p_tsdf = (const float *)tsdf_.data();
    // const float *p_weight = (const float *)weight_.data();
    // const float *p_color = (const float *)color_.data();
    for (int x = 0; x < resolution_; x++) {
        for (int y = 0; y < resolution_; y++) {
            for (int z = 0; z < resolution_; z++) {
                Eigen::Vector3d pt(half_voxel_length + voxel_length_ * x,
                                   half_voxel_length + voxel_length_ * y,
                                   half_voxel_length + voxel_length_ * z);
                int ind = IndexOf(x, y, z);
                if (voxels_[ind].weight_ != 0.0f &&
                    voxels_[ind].tsdf_ < 0.98f &&
                    voxels_[ind].tsdf_ >= -0.98f) {
                    voxel->points_.push_back(pt + origin_);
                    double c = (voxels_[ind].tsdf_ + 1.0) * 0.5;
                    voxel->colors_.push_back(Eigen::Vector3d(c, c, c));
                }
            }
        }
    }
    return voxel;
}

std::shared_ptr<geometry::VoxelGrid> UniformTSDFVolume::ExtractVoxelGrid()
        const {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->voxel_size_ = voxel_length_;
    voxel_grid->origin_ = origin_;

    for (int x = 0; x < resolution_; x++) {
        for (int y = 0; y < resolution_; y++) {
            for (int z = 0; z < resolution_; z++) {
                const int ind = IndexOf(x, y, z);
                const float w = voxels_[ind].weight_;
                const float f = voxels_[ind].tsdf_;
                if (w != 0.0f && f < 0.98f && f >= -0.98f) {
                    double c = (f + 1.0) * 0.5;
                    Eigen::Vector3d color = Eigen::Vector3d(c, c, c);
                    voxel_grid->voxels_.emplace_back(Eigen::Vector3i(x, y, z),
                                                     color);
                }
            }
        }
    }
    return voxel_grid;
}

void UniformTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic,
        const geometry::Image &depth_to_camera_distance_multiplier) {
    const float fx = static_cast<float>(intrinsic.GetFocalLength().first);
    const float fy = static_cast<float>(intrinsic.GetFocalLength().second);
    const float cx = static_cast<float>(intrinsic.GetPrincipalPoint().first);
    const float cy = static_cast<float>(intrinsic.GetPrincipalPoint().second);
    const Eigen::Matrix4f extrinsic_f = extrinsic.cast<float>();
    const float voxel_length_f = static_cast<float>(voxel_length_);
    const float half_voxel_length_f = voxel_length_f * 0.5f;
    const float sdf_trunc_f = static_cast<float>(sdf_trunc_);
    const float sdf_trunc_inv_f = 1.0f / sdf_trunc_f;
    const Eigen::Matrix4f extrinsic_scaled_f = extrinsic_f * voxel_length_f;
    const float safe_width_f = intrinsic.width_ - 0.0001f;
    const float safe_height_f = intrinsic.height_ - 0.0001f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int x = 0; x < resolution_; x++) {
        for (int y = 0; y < resolution_; y++) {
            Eigen::Vector4f pt_3d_homo(float(half_voxel_length_f +
                                             voxel_length_f * x + origin_(0)),
                                       float(half_voxel_length_f +
                                             voxel_length_f * y + origin_(1)),
                                       float(half_voxel_length_f + origin_(2)),
                                       1.f);
            Eigen::Vector4f pt_camera = extrinsic_f * pt_3d_homo;
            for (int z = 0; z < resolution_; z++,
                     pt_camera(0) += extrinsic_scaled_f(0, 2),
                     pt_camera(1) += extrinsic_scaled_f(1, 2),
                     pt_camera(2) += extrinsic_scaled_f(2, 2)) {
                // Skip if negative depth after projection
                if (pt_camera(2) <= 0) {
                    continue;
                }
                // Skip if x-y coordinate not in range
                float u_f = pt_camera(0) * fx / pt_camera(2) + cx + 0.5f;
                float v_f = pt_camera(1) * fy / pt_camera(2) + cy + 0.5f;
                if (!(u_f >= 0.0001f && u_f < safe_width_f && v_f >= 0.0001f &&
                      v_f < safe_height_f)) {
                    continue;
                }
                // Skip if negative depth in depth image
                int u = (int)u_f;
                int v = (int)v_f;
                float d = *image.depth_.PointerAt<float>(u, v);
                if (d <= 0.0f) {
                    continue;
                }

                int v_ind = IndexOf(x, y, z);
                float sdf =
                        (d - pt_camera(2)) *
                        (*depth_to_camera_distance_multiplier.PointerAt<float>(
                                u, v));
                if (sdf > -sdf_trunc_f) {
                    // integrate
                    float tsdf = std::min(1.0f, sdf * sdf_trunc_inv_f);
                    voxels_[v_ind].tsdf_ =
                            (voxels_[v_ind].tsdf_ * voxels_[v_ind].weight_ +
                             tsdf) /
                            (voxels_[v_ind].weight_ + 1.0f);
                    if (color_type_ == TSDFVolumeColorType::RGB8) {
                        const uint8_t *rgb =
                                image.color_.PointerAt<uint8_t>(u, v, 0);
                        Eigen::Vector3d rgb_f(rgb[0], rgb[1], rgb[2]);
                        voxels_[v_ind].color_ =
                                (voxels_[v_ind].color_ *
                                         voxels_[v_ind].weight_ +
                                 rgb_f) /
                                (voxels_[v_ind].weight_ + 1.0f);
                    } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                        const float *intensity =
                                image.color_.PointerAt<float>(u, v, 0);
                        voxels_[v_ind].color_ =
                                (voxels_[v_ind].color_.array() *
                                         voxels_[v_ind].weight_ +
                                 (*intensity)) /
                                (voxels_[v_ind].weight_ + 1.0f);
                    }
                    voxels_[v_ind].weight_ += 1.0f;
                }
            }
        }
    }
}

Eigen::Vector3d UniformTSDFVolume::GetNormalAt(const Eigen::Vector3d &p) {
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

double UniformTSDFVolume::GetTSDFAt(const Eigen::Vector3d &p) {
    Eigen::Vector3i idx;
    Eigen::Vector3d p_grid = p / voxel_length_ - Eigen::Vector3d(0.5, 0.5, 0.5);
    for (int i = 0; i < 3; i++) {
        idx(i) = (int)std::floor(p_grid(i));
    }
    Eigen::Vector3d r = p_grid - idx.cast<double>();

    double tsdf = 0;
    tsdf += (1 - r(0)) * (1 - r(1)) * (1 - r(2)) *
            voxels_[IndexOf(idx + Eigen::Vector3i(0, 0, 0))].tsdf_;
    tsdf += (1 - r(0)) * (1 - r(1)) * r(2) *
            voxels_[IndexOf(idx + Eigen::Vector3i(0, 0, 1))].tsdf_;
    tsdf += (1 - r(0)) * r(1) * (1 - r(2)) *
            voxels_[IndexOf(idx + Eigen::Vector3i(0, 1, 0))].tsdf_;
    tsdf += (1 - r(0)) * r(1) * r(2) *
            voxels_[IndexOf(idx + Eigen::Vector3i(0, 1, 1))].tsdf_;
    tsdf += r(0) * (1 - r(1)) * (1 - r(2)) *
            voxels_[IndexOf(idx + Eigen::Vector3i(1, 0, 0))].tsdf_;
    tsdf += r(0) * (1 - r(1)) * r(2) *
            voxels_[IndexOf(idx + Eigen::Vector3i(1, 0, 1))].tsdf_;
    tsdf += r(0) * r(1) * (1 - r(2)) *
            voxels_[IndexOf(idx + Eigen::Vector3i(1, 1, 0))].tsdf_;
    tsdf += r(0) * r(1) * r(2) *
            voxels_[IndexOf(idx + Eigen::Vector3i(1, 1, 1))].tsdf_;
    return tsdf;
}

}  // namespace integration
}  // namespace open3d
