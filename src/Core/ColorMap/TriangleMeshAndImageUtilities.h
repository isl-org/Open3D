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
#include <Core/Utility/Eigen.h>

namespace open3d {

class Image;
class RGBDImage;
class TriangleMesh;
class ImageWarpingField;
class ColorMapOptimizationOption;
class PinholeCameraTrajectory;

inline std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const PinholeCameraTrajectory& camera, int camid);

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
        CreateVertexAndImageVisibility(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_rgbd,
        const std::vector<std::shared_ptr<Image>>& images_mask,
        const PinholeCameraTrajectory& camera,
        double maximum_allowable_depth,
        double depth_threshold_for_visiblity_check);

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid,
        int ch = -1, int image_boundary_margin = 10);

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const ImageWarpingField& field,
        const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid,
        int ch = -1, int image_boundary_margin = 10);

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<ImageWarpingField>& warping_field,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity, int image_boundary_margin);

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity, int image_boundary_margin);

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_rgbd,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        int image_boundary_margin = 10);

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_rgbd,
        const std::vector<ImageWarpingField>& warping_fields,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        int image_boundary_margin = 10);

}   // namespace open3d
