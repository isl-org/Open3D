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

#include "Open3D/Visualization/Visualizer/RenderOptionWithEditing.h"

#include <json/json.h>

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace visualization {

const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_MIN = 0.000625;
const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_MAX = 0.08;
const double RenderOptionWithEditing::PICKER_SPHERE_SIZE_DEFAULT = 0.01;

bool RenderOptionWithEditing::ConvertToJsonValue(Json::Value &value) const {
    if (RenderOption::ConvertToJsonValue(value) == false) {
        return false;
    }
    if (EigenVector3dToJsonArray(selection_polygon_boundary_color_,
                                 value["selection_polygon_boundary_color"]) ==
        false) {
        return false;
    }
    if (EigenVector3dToJsonArray(selection_polygon_mask_color_,
                                 value["selection_polygon_mask_color"]) ==
        false) {
        return false;
    }
    value["selection_polygon_mask_alpha"] = selection_polygon_mask_alpha_;
    value["pointcloud_picker_sphere_size"] = pointcloud_picker_sphere_size_;
    return true;
}

bool RenderOptionWithEditing::ConvertFromJsonValue(const Json::Value &value) {
    if (RenderOption::ConvertFromJsonValue(value) == false) {
        return false;
    }
    if (EigenVector3dFromJsonArray(selection_polygon_boundary_color_,
                                   value["selection_polygon_boundary_color"]) ==
        false) {
        return false;
    }
    if (EigenVector3dFromJsonArray(selection_polygon_mask_color_,
                                   value["selection_polygon_mask_color"]) ==
        false) {
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
