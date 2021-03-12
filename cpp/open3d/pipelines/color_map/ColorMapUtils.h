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

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace pipelines {
namespace color_map {

std::tuple<std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>,
           std::vector<geometry::Image>>
CreateUtilImagesFromRGBD(const std::vector<geometry::RGBDImage>& images_rgbd);

std::vector<geometry::Image> CreateDepthBoundaryMasks(
        const std::vector<geometry::Image>& images_depth,
        double depth_threshold_for_discontinuity_check,
        int half_dilation_kernel_size_for_discontinuity_map);

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
CreateVertexAndImageVisibility(
        const geometry::TriangleMesh& mesh,
        const std::vector<geometry::Image>& images_depth,
        const std::vector<geometry::Image>& images_mask,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        double maximum_allowable_depth,
        double depth_threshold_for_visibility_check);

void SetProxyIntensityForVertex(
        const geometry::TriangleMesh& mesh,
        const std::vector<geometry::Image>& images_gray,
        const utility::optional<std::vector<ImageWarpingField>>& warping_fields,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        std::vector<double>& proxy_intensity,
        int image_boundary_margin);

void SetGeometryColorAverage(
        geometry::TriangleMesh& mesh,
        const std::vector<geometry::Image>& images_color,
        const utility::optional<std::vector<ImageWarpingField>>& warping_fields,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        int image_boundary_margin = 10,
        int invisible_vertex_color_knn = 3);

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
