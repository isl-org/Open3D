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

#include "open3d/pipelines/color_map/RigidOptimizer.h"

#include <memory>
#include <vector>

#include "open3d/pipelines/color_map/ColorMapUtils.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace pipelines {
namespace color_map {

/// Function to compute i-th row of J and r
/// the vector form of J_r is basically 6x1 matrix, but it can be
/// easily extendable to 6xn matrix.
/// See RGBDOdometryJacobianFromHybridTerm for this case.
static void ComputeJacobianAndResidualRigid(
        int row,
        Eigen::Vector6d& J_r,
        double& r,
        double& w,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<geometry::Image>& images_gray,
        const std::shared_ptr<geometry::Image>& images_dx,
        const std::shared_ptr<geometry::Image>& images_dy,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visibility_image_to_vertex,
        const int image_boundary_margin) {
    J_r.setZero();
    r = 0;
    int vid = visibility_image_to_vertex[row];
    Eigen::Vector3d x = mesh.vertices_[vid];
    Eigen::Vector4d g = extrinsic * Eigen::Vector4d(x(0), x(1), x(2), 1);
    Eigen::Vector4d uv = intrinsic * g;
    double u = uv(0) / uv(2);
    double v = uv(1) / uv(2);
    if (!images_gray->TestImageBoundary(u, v, image_boundary_margin)) return;
    bool valid;
    double gray, dIdx, dIdy;
    std::tie(valid, gray) = images_gray->FloatValueAt(u, v);
    std::tie(valid, dIdx) = images_dx->FloatValueAt(u, v);
    std::tie(valid, dIdy) = images_dy->FloatValueAt(u, v);
    if (gray == -1.0) return;
    double invz = 1. / g(2);
    double v0 = dIdx * intrinsic(0, 0) * invz;
    double v1 = dIdy * intrinsic(1, 1) * invz;
    double v2 = -(v0 * g(0) + v1 * g(1)) * invz;
    J_r(0) = (-g(2) * v1 + g(1) * v2);
    J_r(1) = (g(2) * v0 - g(0) * v2);
    J_r(2) = (-g(1) * v0 + g(0) * v1);
    J_r(3) = v0;
    J_r(4) = v1;
    J_r(5) = v2;
    r = (gray - proxy_intensity[vid]);
    w = 1.0;  // Dummy.
}

void RigidOptimizer::Run(const RigidOptimizerOption& option) {
    utility::LogDebug("[ColorMapOptimization] :: MakingMasks");
    std::vector<std::shared_ptr<geometry::Image>> images_mask =
            CreateDepthBoundaryMasks(
                    images_depth_,
                    option.depth_threshold_for_discontinuity_check_,
                    option.half_dilation_kernel_size_for_discontinuity_map_);

    utility::LogDebug("[ColorMapOptimization] :: VisibilityCheck");
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    *mesh_, images_depth_, images_mask, *camera_trajectory_,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    utility::LogDebug("[ColorMapOptimization] :: Run Rigid Optimization");
    std::vector<double> proxy_intensity;
    int total_num_ = 0;
    int n_camera = int(camera_trajectory_->parameters_.size());
    SetProxyIntensityForVertex(*mesh_, images_gray_, utility::nullopt,
                               *camera_trajectory_, visibility_vertex_to_image,
                               proxy_intensity, option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = camera_trajectory_->parameters_[c].extrinsic_;

            auto intrinsic = camera_trajectory_->parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera_trajectory_->parameters_[c].extrinsic_;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector6d& J_r, double& r,
                                double& w) {
                w = 1.0;  // Dummy.
                ComputeJacobianAndResidualRigid(
                        i, J_r, r, w, *mesh_, proxy_intensity, images_gray_[c],
                        images_dx_[c], images_dy_[c], intr, extrinsic,
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
            camera_trajectory_->parameters_[c].extrinsic_ = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                total_num_ += int(visibility_image_to_vertex[c].size());
            }
        }
        utility::LogDebug("Residual error : {:.6f} (avg : {:.6f})", residual,
                          residual / total_num_);
        SetProxyIntensityForVertex(*mesh_, images_gray_, utility::nullopt,
                                   *camera_trajectory_,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);
    }

    utility::LogDebug("[ColorMapOptimization] :: Set Mesh Color");
    SetGeometryColorAverage(*mesh_, images_color_, utility::nullopt,
                            *camera_trajectory_, visibility_vertex_to_image,
                            option.image_boundary_margin_,
                            option.invisible_vertex_color_knn_);
}

std::shared_ptr<geometry::TriangleMesh> RunRigidOptimizer(
        const geometry::TriangleMesh& mesh,
        const std::vector<std::shared_ptr<geometry::RGBDImage>>& images_rgbd,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const RigidOptimizerOption& option) {
    std::shared_ptr<geometry::TriangleMesh> mesh_;
    std::vector<std::shared_ptr<geometry::RGBDImage>> images_rgbd_;
    std::shared_ptr<camera::PinholeCameraTrajectory> camera_trajectory_;
    std::vector<std::shared_ptr<geometry::Image>> images_gray_;
    std::vector<std::shared_ptr<geometry::Image>> images_dx_;
    std::vector<std::shared_ptr<geometry::Image>> images_dy_;
    std::vector<std::shared_ptr<geometry::Image>> images_color_;
    std::vector<std::shared_ptr<geometry::Image>> images_depth_;
    mesh_ = std::make_shared<geometry::TriangleMesh>(mesh);
    images_rgbd_ = images_rgbd;
    camera_trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>(
            camera_trajectory);

    // images_gray_, images_dx_, images_dy_, images_color_, images_depth_
    // remain unachanged through out the optimizations.
    utility::LogDebug("[ColorMapOptimization] :: CreateGradientImages");
    for (size_t i = 0; i < images_rgbd_.size(); i++) {
        auto gray_image = images_rgbd_[i]->color_.CreateFloatImage();
        auto gray_image_filtered =
                gray_image->Filter(geometry::Image::FilterType::Gaussian3);
        images_gray_.push_back(gray_image_filtered);
        images_dx_.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dx));
        images_dy_.push_back(gray_image_filtered->Filter(
                geometry::Image::FilterType::Sobel3Dy));
        auto color = std::make_shared<geometry::Image>(images_rgbd_[i]->color_);
        auto depth = std::make_shared<geometry::Image>(images_rgbd_[i]->depth_);
        images_color_.push_back(color);
        images_depth_.push_back(depth);
    }

    RigidOptimizer optimizer(mesh_, images_rgbd_, camera_trajectory_,
                             images_gray_, images_dx_, images_dy_,
                             images_color_, images_depth_);
    optimizer.Run(option);
    return mesh_;
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
