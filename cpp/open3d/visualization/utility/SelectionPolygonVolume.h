// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include "open3d/utility/IJsonConvertible.h"

namespace open3d {

namespace geometry {
class Geometry;
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace visualization {

/// \class SelectionPolygonVolume
///
/// \brief Select a polygon volume for cropping.
class SelectionPolygonVolume : public utility::IJsonConvertible {
public:
    ~SelectionPolygonVolume() override {}

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;
    /// Function to crop point cloud.
    ///
    /// \param input The input point cloud.
    std::shared_ptr<geometry::PointCloud> CropPointCloud(
            const geometry::PointCloud &input) const;
    /// Function to crop crop triangle mesh.
    ///
    /// \param input The input triangle mesh.
    std::shared_ptr<geometry::TriangleMesh> CropTriangleMesh(
            const geometry::TriangleMesh &input) const;
    /// Function to crop point cloud with polygon boundaries
    ///
    /// \param input The input point Cloud.
    std::vector<size_t> CropInPolygon(const geometry::PointCloud &input) const;

private:
    std::shared_ptr<geometry::PointCloud> CropPointCloudInPolygon(
            const geometry::PointCloud &input) const;
    std::shared_ptr<geometry::TriangleMesh> CropTriangleMeshInPolygon(
            const geometry::TriangleMesh &input) const;
    std::vector<size_t> CropInPolygon(
            const std::vector<Eigen::Vector3d> &input) const;

public:
    /// One of `{x, y, z}`.
    std::string orthogonal_axis_ = "";
    /// Bounding polygon boundary.
    std::vector<Eigen::Vector3d> bounding_polygon_;
    /// Minimum axis value.
    double axis_min_ = 0.0;
    /// Maximum axis value.
    double axis_max_ = 0.0;
};

}  // namespace visualization
}  // namespace open3d
