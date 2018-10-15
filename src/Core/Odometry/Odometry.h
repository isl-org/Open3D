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

#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <Core/Utility/Console.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Odometry/OdometryOption.h>
#include <Core/Odometry/RGBDOdometryJacobian.h>
#include <Core/Camera/PinholeCameraIntrinsic.h>
#include <Core/Utility/Eigen.h>

namespace open3d {

std::tuple<std::shared_ptr<Image>, std::shared_ptr<Image>>
        InitializeCorrespondenceMap(int width, int height);

void AddElementToCorrespondenceMap(
        Image &correspondence_map, Image &depth_buffer,
        int u_s, int v_s, int u_t, int v_t, float transformed_d_t);

void MergeCorrespondenceMaps(
        Image &correspondence_map, Image &depth_buffer,
        Image &correspondence_map_part, Image &depth_buffer_part);

int CountCorrespondence(const Image &correspondence_map);

std::shared_ptr<CorrespondenceSetPixelWise> ComputeCorrespondence(
        const Eigen::Matrix3d intrinsic_matrix,
        const Eigen::Matrix4d &extrinsic,
        const Image &depth_s, const Image &depth_t,
        const OdometryOption &option);

std::shared_ptr<Image> ConvertDepthImageToXYZImage(
        const Image &depth, const Eigen::Matrix3d &intrinsic_matrix);

std::vector<Eigen::Matrix3d>
        CreateCameraMatrixPyramid(
        const PinholeCameraIntrinsic &pinhole_camera_intrinsic, int levels);

Eigen::Matrix6d CreateInformationMatrix(
        const Eigen::Matrix4d &extrinsic,
        const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Image &depth_s, const Image &depth_t,
        const OdometryOption &option);

void NormalizeIntensity(Image &image_s, Image &image_t,
        CorrespondenceSetPixelWise &correspondence);

std::shared_ptr<RGBDImage> PackRGBDImage(
    const Image &color, const Image &depth);

std::shared_ptr<Image> PreprocessDepth(
        const Image &depth_orig, const OdometryOption &option);

bool CheckImagePair(const Image &image_s, const Image &image_t);

bool CheckRGBDImagePair(const RGBDImage &source, const RGBDImage &target);

std::tuple<std::shared_ptr<RGBDImage>, std::shared_ptr<RGBDImage>>
        InitializeRGBDOdometry(
        const RGBDImage &source, const RGBDImage &target,
        const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4d &odo_init,
        const OdometryOption &option);

std::tuple<bool, Eigen::Matrix4d> DoSingleIteration(
    int iter, int level,
    const RGBDImage &source, const RGBDImage &target,
    const Image &source_xyz,
    const RGBDImage &target_dx, const RGBDImage &target_dy,
    const Eigen::Matrix3d intrinsic,
    const Eigen::Matrix4d &extrinsic_initial,
    const RGBDOdometryJacobian &jacobian_method,
    const OdometryOption &option);

std::tuple<bool, Eigen::Matrix4d> ComputeMultiscale(
        const RGBDImage &source, const RGBDImage &target,
        const PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4d &extrinsic_initial,
        const RGBDOdometryJacobian &jacobian_method,
        const OdometryOption &option);

/// Function to estimate 6D odometry between two RGB-D images
/// output: is_success, 4x4 motion matrix, 6x6 information matrix
std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
        ComputeRGBDOdometry(const RGBDImage &source, const RGBDImage &target,
        const PinholeCameraIntrinsic &pinhole_camera_intrinsic =
        PinholeCameraIntrinsic(),
        const Eigen::Matrix4d &odo_init = Eigen::Matrix4d::Identity(),
        const RGBDOdometryJacobian &jacobian_method =
        RGBDOdometryJacobianFromHybridTerm(),
        const OdometryOption &option = OdometryOption());

}    // namespace open3d
