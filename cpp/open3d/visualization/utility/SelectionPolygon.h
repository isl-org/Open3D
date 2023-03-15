// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "open3d/geometry/Geometry2D.h"
#include "open3d/geometry/Image.h"

namespace open3d {

namespace geometry {
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace visualization {
class ViewControl;
class SelectionPolygonVolume;

/// A 2D polygon used for selection on screen
/// It is a utility class for Visualization
/// The coordinates in SelectionPolygon are lower-left corner based (the OpenGL
/// convention).
class SelectionPolygon : public geometry::Geometry2D {
public:
    enum class SectionPolygonType {
        Unfilled = 0,
        Rectangle = 1,
        Polygon = 2,
    };

public:
    SelectionPolygon()
        : geometry::Geometry2D(geometry::Geometry::GeometryType::Unspecified) {}
    ~SelectionPolygon() override {}

public:
    SelectionPolygon &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2d GetMinBound() const final;
    Eigen::Vector2d GetMaxBound() const final;
    void FillPolygon(int width, int height);
    std::shared_ptr<geometry::PointCloud> CropPointCloud(
            const geometry::PointCloud &input, const ViewControl &view);
    std::shared_ptr<geometry::TriangleMesh> CropTriangleMesh(
            const geometry::TriangleMesh &input, const ViewControl &view);
    std::shared_ptr<SelectionPolygonVolume> CreateSelectionPolygonVolume(
            const ViewControl &view);

private:
    std::shared_ptr<geometry::PointCloud> CropPointCloudInRectangle(
            const geometry::PointCloud &input, const ViewControl &view);
    std::shared_ptr<geometry::PointCloud> CropPointCloudInPolygon(
            const geometry::PointCloud &input, const ViewControl &view);
    std::shared_ptr<geometry::TriangleMesh> CropTriangleMeshInRectangle(
            const geometry::TriangleMesh &input, const ViewControl &view);
    std::shared_ptr<geometry::TriangleMesh> CropTriangleMeshInPolygon(
            const geometry::TriangleMesh &input, const ViewControl &view);
    std::vector<size_t> CropInRectangle(
            const std::vector<Eigen::Vector3d> &input, const ViewControl &view);
    std::vector<size_t> CropInPolygon(const std::vector<Eigen::Vector3d> &input,
                                      const ViewControl &view);

public:
    std::vector<Eigen::Vector2d> polygon_;
    bool is_closed_ = false;
    geometry::Image polygon_interior_mask_;
    SectionPolygonType polygon_type_ = SectionPolygonType::Unfilled;
};

}  // namespace visualization
}  // namespace open3d
