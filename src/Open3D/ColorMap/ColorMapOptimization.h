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

#include <memory>
#include <vector>

namespace open3d {

namespace geometry {
class TriangleMesh;
}
namespace geometry {
class RGBDImage;
}
namespace geometry {
class Image;
}
namespace camera {
class PinholeCameraTrajectory;
}

namespace color_map {

class ColorMapOptimizationOption {
public:
    ColorMapOptimizationOption(
            // Attention: when you update the defaults, update the docstrings in
            // Python/color_map/color_map.cpp
            bool non_rigid_camera_coordinate = false,
            int number_of_vertical_anchors = 16,
            double non_rigid_anchor_point_weight = 0.316,
            int maximum_iteration = 300,
            double maximum_allowable_depth = 2.5,
            double depth_threshold_for_visiblity_check = 0.03,
            double depth_threshold_for_discontinuity_check = 0.1,
            int half_dilation_kernel_size_for_discontinuity_map = 3,
            int image_boundary_margin = 10,
            int invisible_vertex_color_knn = 3)
        : non_rigid_camera_coordinate_(non_rigid_camera_coordinate),
          number_of_vertical_anchors_(number_of_vertical_anchors),
          non_rigid_anchor_point_weight_(non_rigid_anchor_point_weight),
          maximum_iteration_(maximum_iteration),
          maximum_allowable_depth_(maximum_allowable_depth),
          depth_threshold_for_visiblity_check_(
                  depth_threshold_for_visiblity_check),
          depth_threshold_for_discontinuity_check_(
                  depth_threshold_for_discontinuity_check),
          half_dilation_kernel_size_for_discontinuity_map_(
                  half_dilation_kernel_size_for_discontinuity_map),
          image_boundary_margin_(image_boundary_margin),
          invisible_vertex_color_knn_(invisible_vertex_color_knn) {}
    ~ColorMapOptimizationOption() {}

public:
    bool non_rigid_camera_coordinate_;
    int number_of_vertical_anchors_;
    double non_rigid_anchor_point_weight_;
    int maximum_iteration_;
    double maximum_allowable_depth_;
    double depth_threshold_for_visiblity_check_;
    double depth_threshold_for_discontinuity_check_;
    int half_dilation_kernel_size_for_discontinuity_map_;
    int image_boundary_margin_;
    int invisible_vertex_color_knn_;
};

/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014
void ColorMapOptimization(
        geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& imgs_rgbd,
        camera::PinholeCameraTrajectory& camera,
        const ColorMapOptimizationOption& option =
                ColorMapOptimizationOption());
}  // namespace color_map
}  // namespace open3d
