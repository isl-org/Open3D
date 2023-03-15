// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace integration {

/// \enum TSDFVolumeColorType
///
/// Enum class for TSDFVolumeColorType
enum class TSDFVolumeColorType {
    /// No color.
    NoColor = 0,
    /// 8 bit RGB.
    RGB8 = 1,
    /// 32 bit GrayScale.
    Gray32 = 2,
};

/// \class TSDFVolume
///
/// \brief Base class of the Truncated Signed Distance Function (TSDF) volume.
///
/// This volume is usually used to integrate surface data (e.g., a series of
/// RGB-D images) into a Mesh or PointCloud. The basic technique is presented in
/// the following paper:
///
/// B. Curless and M. Levoy
/// A volumetric method for building complex models from range images
/// In SIGGRAPH, 1996
class TSDFVolume {
public:
    /// \brief Default Constructor.
    ///
    /// \param voxel_length Length of the voxel in meters.
    /// \param sdf_trunc Truncation value for signed distance function (SDF).
    /// \param color_type Color type of the TSDF volume.
    TSDFVolume(double voxel_length,
               double sdf_trunc,
               TSDFVolumeColorType color_type)
        : voxel_length_(voxel_length),
          sdf_trunc_(sdf_trunc),
          color_type_(color_type) {}
    virtual ~TSDFVolume() {}

public:
    /// Function to reset the TSDFVolume.
    virtual void Reset() = 0;

    /// Function to integrate an RGB-D image into the volume.
    virtual void Integrate(const geometry::RGBDImage &image,
                           const camera::PinholeCameraIntrinsic &intrinsic,
                           const Eigen::Matrix4d &extrinsic) = 0;

    /// Function to extract a point cloud with normals.
    virtual std::shared_ptr<geometry::PointCloud> ExtractPointCloud() = 0;

    /// \brief Function to extract a triangle mesh, using the marching cubes
    /// algorithm. (https://en.wikipedia.org/wiki/Marching_cubes)
    virtual std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() = 0;

public:
    /// Length of the voxel in meters.
    double voxel_length_;
    /// Truncation value for signed distance function (SDF).
    double sdf_trunc_;
    /// Color type of the TSDF volume.
    TSDFVolumeColorType color_type_;
};

}  // namespace integration
}  // namespace pipelines
}  // namespace open3d
