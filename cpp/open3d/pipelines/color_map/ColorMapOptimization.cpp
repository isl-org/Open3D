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

#include "open3d/pipelines/color_map/ColorMapOptimization.h"

#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/ImageWarpingFieldIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/pipelines/color_map/ColorMapOptimizationJacobian.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/pipelines/color_map/TriangleMeshAndImageUtilities.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace pipelines {
namespace color_map {

static void OptimizeImageCoorNonrigid(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_gray,
        const std::vector<std::shared_ptr<geometry::Image>>& images_dx,
        const std::vector<std::shared_ptr<geometry::Image>>& images_dy,
        std::vector<ImageWarpingField>& warping_fields,
        const std::vector<ImageWarpingField>& warping_fields_init,
        camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        const std::vector<std::vector<int>>& visibility_image_to_vertex,
        std::vector<double>& proxy_intensity,
        const ColorMapOptimizationOption& option) {
    auto n_vertex = mesh.vertices_.size();
    int n_camera = int(camera.parameters_.size());
    SetProxyIntensityForVertex(mesh, images_gray, warping_fields, camera,
                               visibility_vertex_to_image, proxy_intensity,
                               option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        double residual_reg = 0.0;
#pragma omp parallel for schedule(static)
        for (int c = 0; c < n_camera; c++) {
            int nonrigidval = warping_fields[c].anchor_w_ *
                              warping_fields[c].anchor_h_ * 2;
            double rr_reg = 0.0;

            Eigen::Matrix4d pose;
            pose = camera.parameters_[c].extrinsic_;

            auto intrinsic = camera.parameters_[c].intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera.parameters_[c].extrinsic_;
            ColorMapOptimizationJacobian jac;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector14d& J_r, double& r,
                                Eigen::Vector14i& pattern) {
                jac.ComputeJacobianAndResidualNonRigid(
                        i, J_r, r, pattern, mesh, proxy_intensity,
                        images_gray[c], images_dx[c], images_dy[c],
                        warping_fields[c], warping_fields_init[c], intr,
                        extrinsic, visibility_image_to_vertex[c],
                        option.image_boundary_margin_);
            };
            Eigen::MatrixXd JTJ;
            Eigen::VectorXd JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    ComputeJTJandJTrNonRigid<Eigen::Vector14d, Eigen::Vector14i,
                                             Eigen::MatrixXd, Eigen::VectorXd>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            nonrigidval, false);

            double weight = option.non_rigid_anchor_point_weight_ *
                            visibility_image_to_vertex[c].size() / n_vertex;
            for (int j = 0; j < nonrigidval; j++) {
                double r = weight * (warping_fields[c].flow_(j) -
                                     warping_fields_init[c].flow_(j));
                JTJ(6 + j, 6 + j) += weight * weight;
                JTr(6 + j) += weight * r;
                rr_reg += r * r;
            }

            bool success;
            Eigen::VectorXd result;
            std::tie(success, result) = utility::SolveLinearSystemPSD(
                    JTJ, -JTr, /*prefer_sparse=*/false,
                    /*check_symmetric=*/false,
                    /*check_det=*/false, /*check_psd=*/false);
            Eigen::Vector6d result_pose;
            result_pose << result.block(0, 0, 6, 1);
            auto delta = utility::TransformVector6dToMatrix4d(result_pose);
            pose = delta * pose;

            for (int j = 0; j < nonrigidval; j++) {
                warping_fields[c].flow_(j) += result(6 + j);
            }
            camera.parameters_[c].extrinsic_ = pose;

#pragma omp critical
            {
                residual += r2;
                residual_reg += rr_reg;
            }
        }
        utility::LogDebug("Residual error : {:.6f}, reg : {:.6f}", residual,
                          residual_reg);
        SetProxyIntensityForVertex(mesh, images_gray, warping_fields, camera,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);
    }
}

static void OptimizeImageCoorRigid(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::Image>>& images_gray,
        const std::vector<std::shared_ptr<geometry::Image>>& images_dx,
        const std::vector<std::shared_ptr<geometry::Image>>& images_dy,
        camera::PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visibility_vertex_to_image,
        const std::vector<std::vector<int>>& visibility_image_to_vertex,
        std::vector<double>& proxy_intensity,
        const ColorMapOptimizationOption& option) {
    int total_num_ = 0;
    int n_camera = int(camera.parameters_.size());
    SetProxyIntensityForVertex(mesh, images_gray, camera,
                               visibility_vertex_to_image, proxy_intensity,
                               option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        total_num_ = 0;
#pragma omp parallel for schedule(static)
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = camera.parameters_[c].extrinsic_;

            auto intrinsic = camera.parameters_[c].intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera.parameters_[c].extrinsic_;
            ColorMapOptimizationJacobian jac;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector6d& J_r, double& r,
                                double& w) {
                jac.ComputeJacobianAndResidualRigid(
                        i, J_r, r, w, mesh, proxy_intensity, images_gray[c],
                        images_dx[c], images_dy[c], intr, extrinsic,
                        visibility_image_to_vertex[c],
                        option.image_boundary_margin_);
            };
            Eigen::Matrix6d JTJ;
            Eigen::Vector6d JTr;
            double r2;
            std::tie(JTJ, JTr, r2) =
                    utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                            f_lambda, int(visibility_image_to_vertex[c].size()),
                            false);

            bool is_success;
            Eigen::Matrix4d delta;
            std::tie(is_success, delta) =
                    utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ,
                                                                         JTr);
            pose = delta * pose;
            camera.parameters_[c].extrinsic_ = pose;
#pragma omp critical
            {
                residual += r2;
                total_num_ += int(visibility_image_to_vertex[c].size());
            }
        }
        utility::LogDebug("Residual error : {:.6f} (avg : {:.6f})", residual,
                          residual / total_num_);
        SetProxyIntensityForVertex(mesh, images_gray, camera,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);
    }
}

static std::tuple<std::vector<std::shared_ptr<geometry::Image>>,
                  std::vector<std::shared_ptr<geometry::Image>>,
                  std::vector<std::shared_ptr<geometry::Image>>,
                  std::vector<std::shared_ptr<geometry::Image>>,
                  std::vector<std::shared_ptr<geometry::Image>>>
CreateGradientImages(
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& images_rgbd) {
    std::vector<std::shared_ptr<geometry::Image>> images_gray;
    std::vector<std::shared_ptr<geometry::Image>> images_dx;
    std::vector<std::shared_ptr<geometry::Image>> images_dy;
    std::vector<std::shared_ptr<geometry::Image>> images_color;
    std::vector<std::shared_ptr<geometry::Image>> images_depth;
    for (size_t i = 0; i < images_rgbd.size(); i++) {
        auto gray_image = images_rgbd[i]->color_.CreateFloatImage();
        auto gray_image_filtered =
                gray_image->Filter(geometry::Image::FilterType::Gaussian3);
        images_gray.push_back(gray_image_filtered);
        images_dx.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dx));
        images_dy.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dy));
        auto color = std::make_shared<geometry::Image>(images_rgbd[i]->color_);
        auto depth = std::make_shared<geometry::Image>(images_rgbd[i]->depth_);
        images_color.push_back(color);
        images_depth.push_back(depth);
    }
    return std::make_tuple(images_gray, images_dx, images_dy, images_color,
                           images_depth);
}

static std::vector<std::shared_ptr<geometry::Image>> CreateDepthBoundaryMasks(
        const std::vector<std::shared_ptr<geometry::Image>>& images_depth,
        const ColorMapOptimizationOption& option) {
    auto n_images = images_depth.size();
    std::vector<std::shared_ptr<geometry::Image>> masks;
    for (size_t i = 0; i < n_images; i++) {
        utility::LogDebug("[MakeDepthMasks] geometry::Image {:d}/{:d}", i,
                          n_images);
        masks.push_back(images_depth[i]->CreateDepthBoundaryMask(
                option.depth_threshold_for_discontinuity_check_,
                option.half_dilation_kernel_size_for_discontinuity_map_));
    }
    return masks;
}

static std::vector<ImageWarpingField> CreateWarpingFields(
        const std::vector<std::shared_ptr<geometry::Image>>& images,
        const ColorMapOptimizationOption& option) {
    std::vector<ImageWarpingField> fields;
    for (size_t i = 0; i < images.size(); i++) {
        int width = images[i]->width_;
        int height = images[i]->height_;
        fields.push_back(ImageWarpingField(width, height,
                                           option.number_of_vertical_anchors_));
    }
    return fields;
}

void ColorMapOptimization(
        geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& images_rgbd,
        camera::PinholeCameraTrajectory& camera,
        const ColorMapOptimizationOption& option
        /* = ColorMapOptimizationOption()*/) {
    utility::LogDebug("[ColorMapOptimization]");
    std::vector<std::shared_ptr<geometry::Image>> images_gray, images_dx,
            images_dy, images_color, images_depth;
    std::tie(images_gray, images_dx, images_dy, images_color, images_depth) =
            CreateGradientImages(images_rgbd);

    utility::LogDebug("[ColorMapOptimization] :: MakingMasks");
    auto images_mask = CreateDepthBoundaryMasks(images_depth, option);

    utility::LogDebug("[ColorMapOptimization] :: VisibilityCheck");
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    mesh, images_depth, images_mask, camera,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    std::vector<double> proxy_intensity;
    if (option.non_rigid_camera_coordinate_) {
        utility::LogDebug("[ColorMapOptimization] :: Non-Rigid Optimization");
        auto warping_uv_ = CreateWarpingFields(images_gray, option);
        auto warping_uv_init_ = CreateWarpingFields(images_gray, option);
        OptimizeImageCoorNonrigid(
                mesh, images_gray, images_dx, images_dy, warping_uv_,
                warping_uv_init_, camera, visibility_vertex_to_image,
                visibility_image_to_vertex, proxy_intensity, option);
        SetGeometryColorAverage(mesh, images_color, warping_uv_, camera,
                                visibility_vertex_to_image,
                                option.image_boundary_margin_,
                                option.invisible_vertex_color_knn_);
    } else {
        utility::LogDebug("[ColorMapOptimization] :: Rigid Optimization");
        OptimizeImageCoorRigid(mesh, images_gray, images_dx, images_dy, camera,
                               visibility_vertex_to_image,
                               visibility_image_to_vertex, proxy_intensity,
                               option);
        SetGeometryColorAverage(mesh, images_color, camera,
                                visibility_vertex_to_image,
                                option.image_boundary_margin_,
                                option.invisible_vertex_color_knn_);
    }
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
