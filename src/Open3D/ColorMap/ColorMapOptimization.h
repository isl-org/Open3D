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

/// \class ColorMapOptimizationOption
///
/// \brief Defines options for color map optimization.
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
    /// Set to `true` to enable non-rigid optimization (optimizing camera
    /// extrinsic params and image wrapping field for color assignment), set to
    /// False to only enable rigid optimization (optimize camera extrinsic
    /// params).
    bool non_rigid_camera_coordinate_;
    /// Number of vertical anchor points for image wrapping field. The number of
    /// horizontal anchor points is computed automatically based on the number
    /// of vertical anchor points. This option is only used when non-rigid
    /// optimization is enabled.
    int number_of_vertical_anchors_;
    /// Additional regularization terms added to non-rigid regularization. A
    /// higher value results gives more conservative updates. If the residual
    /// error does not stably decrease, it is mainly because images are being
    /// bended abruptly. In this case, consider making iteration more
    /// conservative by increasing the value. This option is only used when
    /// non-rigid optimization is enabled.
    double non_rigid_anchor_point_weight_;
    /// Number of iterations for optimization steps.
    int maximum_iteration_;
    /// Parameter to check the visibility of a point. Points with depth larger
    /// than maximum_allowable_depth in a RGB-D will be marked as invisible for
    /// the camera producing that RGB-D image. Select a proper value to include
    /// necessary points while ignoring unwanted points such as the background.
    double maximum_allowable_depth_;
    /// Parameter to check the visibility of a point. When the difference of a
    /// point’s depth value in the RGB-D image and the point’s depth value in
    /// the 3D mesh is greater than depth_threshold_for_visiblity_check, the
    /// point is marked as invisible to the camera producing the RGB-D image.
    double depth_threshold_for_visiblity_check_;
    /// Parameter to check the visibility of a point. It’s often desirable to
    /// ignore points where there is an abrupt change in depth value. First the
    /// depth gradient image is computed, points are considered to be invisible
    /// if the depth gradient magnitude is larger than
    /// depth_threshold_for_discontinuity_check.
    double depth_threshold_for_discontinuity_check_;
    /// Parameter to check the visibility of a point. Related to
    /// depth_threshold_for_discontinuity_check, when boundary points are
    /// detected, dilation is performed to ignore points near the object
    /// boundary. half_dilation_kernel_size_for_discontinuity_map specifies the
    /// half-kernel size for the dilation applied on the visibility mask image.
    int half_dilation_kernel_size_for_discontinuity_map_;
    ///  If a projected 3D point onto a 2D image lies in the image border within
    ///  image_boundary_margin, the 3D point is considered invisible from the
    ///  camera producing the image. This parmeter is not used for visibility
    ///  check, but used when computing the final color assignment after color
    ///  map optimization.
    int image_boundary_margin_;
    ///  If a vertex is invisible from all images, we assign the averaged color
    ///  of the k nearest visible vertices to fill the invisible vertex. Set to
    ///  0 to disable this feature and all invisible vertices will be black.
    int invisible_vertex_color_knn_;
};

/// \brief Function for color mapping of reconstructed scenes via optimization.
///
/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014.
///
/// \param mesh The input geometry mesh.
/// \param imgs_rgbd A list of RGBDImages seen by cameras.
/// \param camera Cameras' parameters.
/// \param option Color map optimization options. Takes the original
/// ColorMapOptimizationOption values by default.
void ColorMapOptimization(
        geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& imgs_rgbd,
        camera::PinholeCameraTrajectory& camera,
        const ColorMapOptimizationOption& option =
                ColorMapOptimizationOption());
}  // namespace color_map
}  // namespace open3d
