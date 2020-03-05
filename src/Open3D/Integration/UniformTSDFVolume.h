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

#pragma once

#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Integration/TSDFVolume.h"

namespace open3d {

namespace geometry {

class TSDFVoxel : public Voxel {
public:
    TSDFVoxel() : Voxel() {}
    TSDFVoxel(const Eigen::Vector3i &grid_index) : Voxel(grid_index) {}
    TSDFVoxel(const Eigen::Vector3i &grid_index, const Eigen::Vector3d &color)
        : Voxel(grid_index, color) {}
    ~TSDFVoxel() {}

public:
    float tsdf_ = 0;
    float weight_ = 0;
};

}  // namespace geometry

namespace integration {

/// \class UniformTSDFVolume
///
/// \brief UniformTSDFVolume implements the classic TSDF volume with uniform
/// voxel grid (Curless and Levoy 1996).
class UniformTSDFVolume : public TSDFVolume {
public:
    UniformTSDFVolume(double length,
                      int resolution,
                      double sdf_trunc,
                      TSDFVolumeColorType color_type,
                      const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());
    ~UniformTSDFVolume() override;

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4d &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;

    /// Debug function to extract the voxel data into a VoxelGrid
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud() const;
    /// Debug function to extract the voxel data VoxelGrid
    std::shared_ptr<geometry::VoxelGrid> ExtractVoxelGrid() const;

    /// Faster Integrate function that uses depth_to_camera_distance_multiplier
    /// precomputed from camera intrinsic
    void IntegrateWithDepthToCameraDistanceMultiplier(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const geometry::Image &depth_to_camera_distance_multiplier);

    inline int IndexOf(int x, int y, int z) const {
        return x * resolution_ * resolution_ + y * resolution_ + z;
    }

    inline int IndexOf(const Eigen::Vector3i &xyz) const {
        return IndexOf(xyz(0), xyz(1), xyz(2));
    }

public:
    std::vector<geometry::TSDFVoxel> voxels_;
    Eigen::Vector3d origin_;
    /// Total length, where voxel_length = length / resolution.
    double length_;
    /// Resolution over the total length, where voxel_length = length /
    /// resolution.
    int resolution_;
    /// Number of voxels present.
    int voxel_num_;

private:
    Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

    double GetTSDFAt(const Eigen::Vector3d &p);
};

}  // namespace integration
}  // namespace open3d
