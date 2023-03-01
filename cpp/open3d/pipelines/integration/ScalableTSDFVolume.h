// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <unordered_map>

#include "open3d/pipelines/integration/TSDFVolume.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace pipelines {
namespace integration {

class UniformTSDFVolume;

/// The ScalableTSDFVolume implements a more memory efficient data structure for
/// volumetric integration.
///
/// This implementation is based on the following repository:
/// https://github.com/qianyizh/ElasticReconstruction/tree/master/Integrate
/// The reference is:
/// Q.-Y. Zhou and V. Koltun
/// Dense Scene Reconstruction with Points of Interest
/// In SIGGRAPH 2013
///
/// An observed depth pixel gives two types of information: (a) an approximation
/// of the nearby surface, and (b) empty space from the camera to the surface.
/// They induce two core concepts of volumetric integration: weighted average of
/// a truncated signed distance function (TSDF), and carving. The weighted
/// average of TSDF is great in addressing the Gaussian noise along surface
/// normal and producing a smooth surface output. The carving is great in
/// removing outlier structures like floating noise pixels and bumps along
/// structure edges.
class ScalableTSDFVolume : public TSDFVolume {
public:
    struct VolumeUnit {
    public:
        VolumeUnit() : volume_(NULL) {}

    public:
        std::shared_ptr<UniformTSDFVolume> volume_;
        Eigen::Vector3i index_;
    };

public:
    ScalableTSDFVolume(double voxel_length,
                       double sdf_trunc,
                       TSDFVolumeColorType color_type,
                       int volume_unit_resolution = 16,
                       int depth_sampling_stride = 4);
    ~ScalableTSDFVolume() override;

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4d &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;
    /// Debug function to extract the voxel data into a point cloud.
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud();

public:
    int volume_unit_resolution_;
    double volume_unit_length_;
    int depth_sampling_stride_;

    /// Assume the index of the volume unit is (x, y, z), then the unit spans
    /// from (x, y, z) * volume_unit_length_
    /// to (x + 1, y + 1, z + 1) * volume_unit_length_
    std::unordered_map<Eigen::Vector3i,
                       VolumeUnit,
                       utility::hash_eigen<Eigen::Vector3i>>
            volume_units_;

private:
    Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3d &point) {
        return Eigen::Vector3i((int)std::floor(point(0) / volume_unit_length_),
                               (int)std::floor(point(1) / volume_unit_length_),
                               (int)std::floor(point(2) / volume_unit_length_));
    }

    std::shared_ptr<UniformTSDFVolume> OpenVolumeUnit(
            const Eigen::Vector3i &index);

    Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

    double GetTSDFAt(const Eigen::Vector3d &p);
};

}  // namespace integration
}  // namespace pipelines
}  // namespace open3d
