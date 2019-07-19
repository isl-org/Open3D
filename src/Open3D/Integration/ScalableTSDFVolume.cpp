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

#include "Open3D/Integration/ScalableTSDFVolume.h"

#include <unordered_set>

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Integration/MarchingCubesConst.h"
#include "Open3D/Integration/UniformTSDFVolume.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace integration {

ScalableTSDFVolume::ScalableTSDFVolume(double voxel_length,
                                       double sdf_trunc,
                                       TSDFVolumeColorType color_type,
                                       int volume_unit_resolution /* = 16*/,
                                       int depth_sampling_stride /* = 4*/)
    : TSDFVolume(voxel_length, sdf_trunc, color_type),
      volume_unit_resolution_(volume_unit_resolution),
      volume_unit_length_(voxel_length * volume_unit_resolution),
      depth_sampling_stride_(depth_sampling_stride) {}

ScalableTSDFVolume::~ScalableTSDFVolume() {}

void ScalableTSDFVolume::Reset() { volume_units_.clear(); }

void ScalableTSDFVolume::Integrate(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4d &extrinsic) {
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
                "[ScalableTSDFVolume::Integrate] Unsupported image format.\n");
        return;
    }
    auto depth2cameradistance =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);
    auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
            image.depth_, intrinsic, extrinsic, 1000.0, 1000.0,
            depth_sampling_stride_);
    std::unordered_set<Eigen::Vector3i,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            touched_volume_units_;
    for (const auto &point : pointcloud->points_) {
        auto min_bound = LocateVolumeUnit(
                point - Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
        auto max_bound = LocateVolumeUnit(
                point + Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
        for (auto x = min_bound(0); x <= max_bound(0); x++) {
            for (auto y = min_bound(1); y <= max_bound(1); y++) {
                for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (touched_volume_units_.find(loc) ==
                        touched_volume_units_.end()) {
                        touched_volume_units_.insert(loc);
                        auto volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
                        volume->IntegrateWithDepthToCameraDistanceMultiplier(
                                image, intrinsic, extrinsic,
                                *depth2cameradistance);
                    }
                }
            }
        }
    }
}

std::shared_ptr<geometry::PointCloud> ScalableTSDFVolume::ExtractPointCloud() {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    double half_voxel_length = voxel_length_ * 0.5;
    float w0, w1, f0, f1;
    Eigen::Vector3f c0, c1;
    for (const auto &unit : volume_units_) {
        if (unit.second.volume_) {
            const auto &volume0 = *unit.second.volume_;
            const auto &index0 = unit.second.index_;
            for (int x = 0; x < volume0.resolution_; x++) {
                for (int y = 0; y < volume0.resolution_; y++) {
                    for (int z = 0; z < volume0.resolution_; z++) {
                        Eigen::Vector3i idx0(x, y, z);
                        w0 = volume0.voxels_[volume0.IndexOf(idx0)].weight_;
                        f0 = volume0.voxels_[volume0.IndexOf(idx0)].tsdf_;
                        if (color_type_ != TSDFVolumeColorType::None)
                            c0 = volume0.voxels_[volume0.IndexOf(idx0)]
                                         .color_.cast<float>();
                        if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f) {
                            Eigen::Vector3d p0 =
                                    Eigen::Vector3d(half_voxel_length +
                                                            voxel_length_ * x,
                                                    half_voxel_length +
                                                            voxel_length_ * y,
                                                    half_voxel_length +
                                                            voxel_length_ * z) +
                                    index0.cast<double>() * volume_unit_length_;
                            for (int i = 0; i < 3; i++) {
                                Eigen::Vector3d p1 = p0;
                                Eigen::Vector3i idx1 = idx0;
                                Eigen::Vector3i index1 = index0;
                                p1(i) += voxel_length_;
                                idx1(i) += 1;
                                if (idx1(i) < volume0.resolution_) {
                                    w1 = volume0.voxels_[volume0.IndexOf(idx1)]
                                                 .weight_;
                                    f1 = volume0.voxels_[volume0.IndexOf(idx1)]
                                                 .tsdf_;
                                    if (color_type_ !=
                                        TSDFVolumeColorType::None)
                                        c1 = volume0.voxels_[volume0.IndexOf(
                                                                     idx1)]
                                                     .color_.cast<float>();
                                } else {
                                    idx1(i) -= volume0.resolution_;
                                    index1(i) += 1;
                                    auto unit_itr = volume_units_.find(index1);
                                    if (unit_itr == volume_units_.end()) {
                                        w1 = 0.0f;
                                        f1 = 0.0f;
                                    } else {
                                        const auto &volume1 =
                                                *unit_itr->second.volume_;
                                        w1 = volume1.voxels_[volume1.IndexOf(
                                                                     idx1)]
                                                     .weight_;
                                        f1 = volume1.voxels_[volume1.IndexOf(
                                                                     idx1)]
                                                     .tsdf_;
                                        if (color_type_ !=
                                            TSDFVolumeColorType::None)
                                            c1 = volume1.voxels_
                                                         [volume1.IndexOf(idx1)]
                                                                 .color_
                                                                 .cast<float>();
                                    }
                                }
                                if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
                                    f0 * f1 < 0) {
                                    float r0 = std::fabs(f0);
                                    float r1 = std::fabs(f1);
                                    Eigen::Vector3d p = p0;
                                    p(i) = (p0(i) * r1 + p1(i) * r0) /
                                           (r0 + r1);
                                    pointcloud->points_.push_back(p);
                                    if (color_type_ ==
                                        TSDFVolumeColorType::RGB8) {
                                        pointcloud->colors_.push_back(
                                                ((c0 * r1 + c1 * r0) /
                                                 (r0 + r1) / 255.0f)
                                                        .cast<double>());
                                    } else if (color_type_ ==
                                               TSDFVolumeColorType::Gray32) {
                                        pointcloud->colors_.push_back(
                                                ((c0 * r1 + c1 * r0) /
                                                 (r0 + r1))
                                                        .cast<double>());
                                    }
                                    // has_normal
                                    pointcloud->normals_.push_back(
                                            GetNormalAt(p));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh>
ScalableTSDFVolume::ExtractTriangleMesh() {
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    double half_voxel_length = voxel_length_ * 0.5;
    std::unordered_map<
            Eigen::Vector4i, int, utility::hash_eigen::hash<Eigen::Vector4i>,
            std::equal_to<Eigen::Vector4i>,
            Eigen::aligned_allocator<std::pair<const Eigen::Vector4i, int>>>
            edgeindex_to_vertexindex;
    int edge_to_index[12];
    for (const auto &unit : volume_units_) {
        if (unit.second.volume_) {
            const auto &volume0 = *unit.second.volume_;
            const auto &index0 = unit.second.index_;
            for (int x = 0; x < volume0.resolution_; x++) {
                for (int y = 0; y < volume0.resolution_; y++) {
                    for (int z = 0; z < volume0.resolution_; z++) {
                        Eigen::Vector3i idx0(x, y, z);
                        int cube_index = 0;
                        float w[8];
                        float f[8];
                        Eigen::Vector3d c[8];
                        for (int i = 0; i < 8; i++) {
                            Eigen::Vector3i index1 = index0;
                            Eigen::Vector3i idx1 = idx0 + shift[i];
                            if (idx1(0) < volume_unit_resolution_ &&
                                idx1(1) < volume_unit_resolution_ &&
                                idx1(2) < volume_unit_resolution_) {
                                w[i] = volume0.voxels_[volume0.IndexOf(idx1)]
                                               .weight_;
                                f[i] = volume0.voxels_[volume0.IndexOf(idx1)]
                                               .tsdf_;
                                if (color_type_ == TSDFVolumeColorType::RGB8)
                                    c[i] = volume0.voxels_[volume0.IndexOf(
                                                                   idx1)]
                                                   .color_.cast<double>() /
                                           255.0;
                                else if (color_type_ ==
                                         TSDFVolumeColorType::Gray32)
                                    c[i] = volume0.voxels_[volume0.IndexOf(
                                                                   idx1)]
                                                   .color_.cast<double>();
                            } else {
                                for (int j = 0; j < 3; j++) {
                                    if (idx1(j) >= volume_unit_resolution_) {
                                        idx1(j) -= volume_unit_resolution_;
                                        index1(j) += 1;
                                    }
                                }
                                auto unit_itr1 = volume_units_.find(index1);
                                if (unit_itr1 == volume_units_.end()) {
                                    w[i] = 0.0f;
                                    f[i] = 0.0f;
                                } else {
                                    const auto &volume1 =
                                            *unit_itr1->second.volume_;
                                    w[i] = volume1.voxels_[volume1.IndexOf(
                                                                   idx1)]
                                                   .weight_;
                                    f[i] = volume1.voxels_[volume1.IndexOf(
                                                                   idx1)]
                                                   .tsdf_;
                                    if (color_type_ ==
                                        TSDFVolumeColorType::RGB8)
                                        c[i] = volume1.voxels_[volume1.IndexOf(
                                                                       idx1)]
                                                       .color_.cast<double>() /
                                               255.0;
                                    else if (color_type_ ==
                                             TSDFVolumeColorType::Gray32)
                                        c[i] = volume1.voxels_[volume1.IndexOf(
                                                                       idx1)]
                                                       .color_.cast<double>();
                                }
                            }
                            if (w[i] == 0.0f) {
                                cube_index = 0;
                                break;
                            } else {
                                if (f[i] < 0.0f) {
                                    cube_index |= (1 << i);
                                }
                            }
                        }
                        if (cube_index == 0 || cube_index == 255) {
                            continue;
                        }
                        for (int i = 0; i < 12; i++) {
                            if (edge_table[cube_index] & (1 << i)) {
                                Eigen::Vector4i edge_index =
                                        Eigen::Vector4i(index0(0), index0(1),
                                                        index0(2), 0) *
                                                volume_unit_resolution_ +
                                        Eigen::Vector4i(x, y, z, 0) +
                                        edge_shift[i];
                                if (edgeindex_to_vertexindex.find(edge_index) ==
                                    edgeindex_to_vertexindex.end()) {
                                    edge_to_index[i] =
                                            (int)mesh->vertices_.size();
                                    edgeindex_to_vertexindex[edge_index] =
                                            (int)mesh->vertices_.size();
                                    Eigen::Vector3d pt(
                                            half_voxel_length +
                                                    voxel_length_ *
                                                            edge_index(0),
                                            half_voxel_length +
                                                    voxel_length_ *
                                                            edge_index(1),
                                            half_voxel_length +
                                                    voxel_length_ *
                                                            edge_index(2));
                                    double f0 = std::abs(
                                            (double)f[edge_to_vert[i][0]]);
                                    double f1 = std::abs(
                                            (double)f[edge_to_vert[i][1]]);
                                    pt(edge_index(3)) +=
                                            f0 * voxel_length_ / (f0 + f1);
                                    mesh->vertices_.push_back(pt);
                                    if (color_type_ !=
                                        TSDFVolumeColorType::None) {
                                        const auto &c0 = c[edge_to_vert[i][0]];
                                        const auto &c1 = c[edge_to_vert[i][1]];
                                        mesh->vertex_colors_.push_back(
                                                (f1 * c0 + f0 * c1) /
                                                (f0 + f1));
                                    }
                                } else {
                                    edge_to_index[i] = edgeindex_to_vertexindex
                                            [edge_index];
                                }
                            }
                        }
                        for (int i = 0; tri_table[cube_index][i] != -1;
                             i += 3) {
                            mesh->triangles_.push_back(Eigen::Vector3i(
                                    edge_to_index[tri_table[cube_index][i]],
                                    edge_to_index[tri_table[cube_index][i + 2]],
                                    edge_to_index[tri_table[cube_index]
                                                           [i + 1]]));
                        }
                    }
                }
            }
        }
    }
    return mesh;
}

std::shared_ptr<geometry::PointCloud>
ScalableTSDFVolume::ExtractVoxelPointCloud() {
    auto voxel = std::make_shared<geometry::PointCloud>();
    for (auto &unit : volume_units_) {
        if (unit.second.volume_) {
            auto v = unit.second.volume_->ExtractVoxelPointCloud();
            *voxel += *v;
        }
    }
    return voxel;
}

std::shared_ptr<UniformTSDFVolume> ScalableTSDFVolume::OpenVolumeUnit(
        const Eigen::Vector3i &index) {
    auto &unit = volume_units_[index];
    if (!unit.volume_) {
        unit.volume_.reset(new UniformTSDFVolume(
                volume_unit_length_, volume_unit_resolution_, sdf_trunc_,
                color_type_, index.cast<double>() * volume_unit_length_));
        unit.index_ = index;
    }
    return unit.volume_;
}

Eigen::Vector3d ScalableTSDFVolume::GetNormalAt(const Eigen::Vector3d &p) {
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

double ScalableTSDFVolume::GetTSDFAt(const Eigen::Vector3d &p) {
    Eigen::Vector3d p_locate =
            p - Eigen::Vector3d(0.5, 0.5, 0.5) * voxel_length_;
    Eigen::Vector3i index0 = LocateVolumeUnit(p_locate);
    auto unit_itr = volume_units_.find(index0);
    if (unit_itr == volume_units_.end()) {
        return 0.0;
    }
    const auto &volume0 = *unit_itr->second.volume_;
    Eigen::Vector3i idx0;
    Eigen::Vector3d p_grid =
            (p_locate - index0.cast<double>() * volume_unit_length_) /
            voxel_length_;
    for (int i = 0; i < 3; i++) {
        idx0(i) = (int)std::floor(p_grid(i));
        if (idx0(i) < 0) idx0(i) = 0;
        if (idx0(i) >= volume_unit_resolution_)
            idx0(i) = volume_unit_resolution_ - 1;
    }
    Eigen::Vector3d r = p_grid - idx0.cast<double>();
    float f[8];
    for (int i = 0; i < 8; i++) {
        Eigen::Vector3i index1 = index0;
        Eigen::Vector3i idx1 = idx0 + shift[i];
        if (idx1(0) < volume_unit_resolution_ &&
            idx1(1) < volume_unit_resolution_ &&
            idx1(2) < volume_unit_resolution_) {
            f[i] = volume0.voxels_[volume0.IndexOf(idx1)].tsdf_;
        } else {
            for (int j = 0; j < 3; j++) {
                if (idx1(j) >= volume_unit_resolution_) {
                    idx1(j) -= volume_unit_resolution_;
                    index1(j) += 1;
                }
            }
            auto unit_itr1 = volume_units_.find(index1);
            if (unit_itr1 == volume_units_.end()) {
                f[i] = 0.0f;
            } else {
                const auto &volume1 = *unit_itr1->second.volume_;
                f[i] = volume1.voxels_[volume1.IndexOf(idx1)].tsdf_;
            }
        }
    }
    return (1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
                         r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
           r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
                   r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));
}

}  // namespace integration
}  // namespace open3d
