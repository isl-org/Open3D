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

#include "ColorMapOptimizationOption.h"
#include "ImageWarpingField.h"

#include <memory>
#include <vector>

namespace open3d {

class TriangleMesh;
class RGBDImage;
class Image;
class PinholeCameraTrajectory;

// const double IMAGE_BOUNDARY_MARGIN = 10;

std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const PinholeCameraTrajectory& camera, int camid);

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
        MakeVertexAndImageVisibility(const TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const std::vector<Image>& images_mask,
        const PinholeCameraTrajectory& camera,
        const ColorMapOptmizationOption& option);

std::vector<ImageWarpingField> MakeWarpingFields(
        const std::vector<std::shared_ptr<Image>>& images,
        const ColorMapOptmizationOption& option);

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid, int ch = -1);

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const ImageWarpingField& field,
        const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid, int ch = -1);

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<ImageWarpingField>& warping_field,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity);

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity);

void OptimizeImageCoorNonrigid(
        const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<std::shared_ptr<Image>>& images_dx,
        const std::vector<std::shared_ptr<Image>>& images_dy,
        std::vector<ImageWarpingField>& warping_fields,
        const std::vector<ImageWarpingField>& warping_fields_init,
        PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        const std::vector<std::vector<int>>& visiblity_image_to_vertex,
        std::vector<double>& proxy_intensity,
        const ColorMapOptmizationOption& option);

void OptimizeImageCoorRigid(
        const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<std::shared_ptr<Image>>& images_dx,
        const std::vector<std::shared_ptr<Image>>& images_dy,
        PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        const std::vector<std::vector<int>>& visiblity_image_to_vertex,
        std::vector<double>& proxy_intensity,
        const ColorMapOptmizationOption& option);

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const std::vector<ImageWarpingField>& warping_fields,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image);

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image);

std::tuple<std::vector<std::shared_ptr<Image>>,
        std::vector<std::shared_ptr<Image>>,
        std::vector<std::shared_ptr<Image>>> MakeGradientImages(
        const std::vector<RGBDImage>& images_rgbd);

std::vector<Image> MakeDepthMasks(
        const std::vector<RGBDImage>& images_rgbd,
        const ColorMapOptmizationOption& option);

/// This is implementation of following paper
/// Q.-Y. Zhou and V. Koltun,
/// Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
/// SIGGRAPH 2014
void ColorMapOptimization(TriangleMesh& mesh,
        const std::vector<RGBDImage>& imgs_rgbd,
        PinholeCameraTrajectory& camera,
        const ColorMapOptmizationOption& option =
        ColorMapOptmizationOption());

}	// namespace open3d
