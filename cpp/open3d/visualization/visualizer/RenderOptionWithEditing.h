// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include "open3d/visualization/visualizer/RenderOption.h"

namespace open3d {
namespace visualization {

class RenderOptionWithEditing : public RenderOption {
public:
    static const double PICKER_SPHERE_SIZE_MIN;
    static const double PICKER_SPHERE_SIZE_MAX;
    static const double PICKER_SPHERE_SIZE_DEFAULT;

public:
    RenderOptionWithEditing() {}
    ~RenderOptionWithEditing() override {}

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;
    void IncreaseSphereSize() {
        pointcloud_picker_sphere_size_ = std::min(
                pointcloud_picker_sphere_size_ * 2.0, PICKER_SPHERE_SIZE_MAX);
    }
    void DecreaseSphereSize() {
        pointcloud_picker_sphere_size_ = std::max(
                pointcloud_picker_sphere_size_ * 0.5, PICKER_SPHERE_SIZE_MIN);
    }

public:
    // Selection polygon
    Eigen::Vector3d selection_polygon_boundary_color_ =
            Eigen::Vector3d(0.3, 0.3, 0.3);
    Eigen::Vector3d selection_polygon_mask_color_ =
            Eigen::Vector3d(0.3, 0.3, 0.3);
    double selection_polygon_mask_alpha_ = 0.5;

    // PointCloud Picker
    double pointcloud_picker_sphere_size_ = PICKER_SPHERE_SIZE_DEFAULT;
};

}  // namespace visualization
}  // namespace open3d
