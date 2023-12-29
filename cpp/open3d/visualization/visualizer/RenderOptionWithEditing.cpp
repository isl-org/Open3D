// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/RenderOptionWithEditing.h"

#include <json/json.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {

const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_MIN = 0.000625;
const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_MAX = 0.08;
const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_DEFAULT = 0.01;

bool RenderOptionWithEditing::ConvertToJsonValue(Json::Value &value) const {
    if (!RenderOption::ConvertToJsonValue(value)) {
        return false;
    }
    if (!EigenVector3dToJsonArray(selection_polygon_boundary_color_,
                                  value["selection_polygon_boundary_color"])) {
        return false;
    }
    if (!EigenVector3dToJsonArray(selection_polygon_mask_color_,
                                  value["selection_polygon_mask_color"])) {
        return false;
    }
    value["selection_polygon_mask_alpha"] = selection_polygon_mask_alpha_;
    value["pointcloud_picker_sphere_size"] = pointcloud_picker_sphere_size_;
    return true;
}

bool RenderOptionWithEditing::ConvertFromJsonValue(const Json::Value &value) {
    if (!RenderOption::ConvertFromJsonValue(value)) {
        return false;
    }
    if (!EigenVector3dFromJsonArray(
                selection_polygon_boundary_color_,
                value["selection_polygon_boundary_color"])) {
        return false;
    }
    if (!EigenVector3dFromJsonArray(selection_polygon_mask_color_,
                                    value["selection_polygon_mask_color"])) {
        return false;
    }
    selection_polygon_mask_alpha_ = value.get("selection_polygon_mask_alpha",
                                              selection_polygon_mask_alpha_)
                                            .asDouble();
    pointcloud_picker_sphere_size_ = value.get("pointcloud_picker_sphere_size",
                                               selection_polygon_mask_alpha_)
                                             .asDouble();
    return true;
}

}  // namespace visualization
}  // namespace open3d
