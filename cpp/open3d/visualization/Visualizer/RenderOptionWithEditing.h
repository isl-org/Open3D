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

#include <Eigen/Core>

#include "Open3D/Visualization/Visualizer/RenderOption.h"

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
