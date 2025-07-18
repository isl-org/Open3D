// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace color_map {

struct RigidOptimizerOption {
    /// Number of iterations for optimization steps.
    int maximum_iteration_ = 300;

    /// Parameter to check the visibility of a point. Points with depth larger
    /// than maximum_allowable_depth in a RGB-D will be marked as invisible for
    /// the camera producing that RGB-D image. Select a proper value to include
    /// necessary points while ignoring unwanted points such as the background.
    double maximum_allowable_depth_ = 2.5;

    /// Parameter to check the visibility of a point. When the difference of a
    /// point’s depth value in the RGB-D image and the point’s depth value in
    /// the 3D mesh is greater than depth_threshold_for_visibility_check, the
    /// point is marked as invisible to the camera producing the RGB-D image.
    double depth_threshold_for_visibility_check_ = 0.03;

    /// Parameter to check the visibility of a point. It’s often desirable to
    /// ignore points where there is an abrupt change in depth value. First the
    /// depth gradient image is computed, points are considered to be invisible
    /// if the depth gradient magnitude is larger than
    /// depth_threshold_for_discontinuity_check.
    double depth_threshold_for_discontinuity_check_ = 0.1;

    /// Parameter to check the visibility of a point. Related to
    /// depth_threshold_for_discontinuity_check, when boundary points are
    /// detected, dilation is performed to ignore points near the object
    /// boundary. half_dilation_kernel_size_for_discontinuity_map specifies the
    /// half-kernel size for the dilation applied on the visibility mask image.
    int half_dilation_kernel_size_for_discontinuity_map_ = 3;

    /// If a projected 3D point onto a 2D image lies in the image border within
    /// image_boundary_margin, the 3D point is considered invisible from the
    /// camera producing the image. This parameter is not used for visibility
    /// check, but used when computing the final color assignment after color
    /// map optimization.
    int image_boundary_margin_ = 10;

    /// If a vertex is invisible from all images, we assign the averaged color
    /// of the k nearest visible vertices to fill the invisible vertex. Set to
    /// 0 to disable this feature and all invisible vertices will be black.
    int invisible_vertex_color_knn_ = 3;

    /// If specified, the intermediate results will be stored in in the debug
    /// output dir. Existing files will be overwritten if the names are the
    /// same.
    std::string debug_output_dir_ = "";
};

std::pair<geometry::TriangleMesh, camera::PinholeCameraTrajectory>
RunRigidOptimizer(const geometry::TriangleMesh& mesh,
                  const std::vector<geometry::RGBDImage>& images_rgbd,
                  const camera::PinholeCameraTrajectory& camera_trajectory,
                  const RigidOptimizerOption& option);

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
