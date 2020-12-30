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

#include "open3d/io/ImageIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/pipelines/color_map/ColorMapUtils.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/utility/FileSystem.h"
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
        const geometry::Image& images_gray,
        const geometry::Image& images_dx,
        const geometry::Image& images_dy,
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
    if (!images_gray.TestImageBoundary(u, v, image_boundary_margin)) return;
    bool valid;
    double gray, dIdx, dIdy;
    std::tie(valid, gray) = images_gray.FloatValueAt(u, v);
    std::tie(valid, dIdx) = images_dx.FloatValueAt(u, v);
    std::tie(valid, dIdy) = images_dy.FloatValueAt(u, v);
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

geometry::TriangleMesh RunRigidOptimizer(
        const geometry::TriangleMesh& mesh,
        const std::vector<geometry::RGBDImage>& images_rgbd,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const RigidOptimizerOption& option) {
    // The following properties will change during optimization.
    geometry::TriangleMesh opt_mesh = mesh;
    camera::PinholeCameraTrajectory opt_camera_trajectory = camera_trajectory;

    // The following properties remain unchanged during optimization.
    std::vector<geometry::Image> images_gray;
    std::vector<geometry::Image> images_dx;
    std::vector<geometry::Image> images_dy;
    std::vector<geometry::Image> images_color;
    std::vector<geometry::Image> images_depth;
    std::vector<geometry::Image> images_mask;
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;

    // Create all debugging directories. We don't delete any existing files but
    // will overwrite them if the names are the same.
    if (!option.debug_output_dir_.empty()) {
        std::vector<std::string> dirs{
                option.debug_output_dir_, option.debug_output_dir_ + "/rigid",
                option.debug_output_dir_ + "/rigid/images_mask",
                option.debug_output_dir_ + "/rigid/opt_mesh",
                option.debug_output_dir_ + "/rigid/opt_camera_trajectory"};
        for (const std::string& dir : dirs) {
            if (utility::filesystem::DirectoryExists(dir)) {
                utility::LogInfo("Directory exists: {}.", dir);
            } else {
                if (utility::filesystem::MakeDirectoryHierarchy(dir)) {
                    utility::LogInfo("Directory created: {}.", dir);
                } else {
                    utility::LogError("Making directory failed: {}.", dir);
                }
            }
        }
    }

    utility::LogDebug("[ColorMapOptimization] CreateUtilImagesFromRGBD");
    std::tie(images_gray, images_dx, images_dy, images_color, images_depth) =
            CreateUtilImagesFromRGBD(images_rgbd);

    utility::LogDebug("[ColorMapOptimization] CreateDepthBoundaryMasks");
    images_mask = CreateDepthBoundaryMasks(
            images_depth, option.depth_threshold_for_discontinuity_check_,
            option.half_dilation_kernel_size_for_discontinuity_map_);
    if (!option.debug_output_dir_.empty()) {
        for (size_t i = 0; i < images_mask.size(); ++i) {
            std::string file_name = fmt::format(
                    "{}/{}.png",
                    option.debug_output_dir_ + "/rigid/images_mask", i);
            io::WriteImage(file_name, images_mask[i]);
        }
    }

    utility::LogDebug("[ColorMapOptimization] CreateVertexAndImageVisibility");
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    opt_mesh, images_depth, images_mask, opt_camera_trajectory,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    utility::LogDebug("[ColorMapOptimization] Rigid Optimization");
    std::vector<double> proxy_intensity;
    int total_num_ = 0;
    int n_camera = int(opt_camera_trajectory.parameters_.size());
    SetProxyIntensityForVertex(opt_mesh, images_gray, utility::nullopt,
                               opt_camera_trajectory,
                               visibility_vertex_to_image, proxy_intensity,
                               option.image_boundary_margin_);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        utility::LogDebug("[Iteration {:04d}] ", itr + 1);
        double residual = 0.0;
        total_num_ = 0;
#pragma omp parallel for schedule(static)
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = opt_camera_trajectory.parameters_[c].extrinsic_;

            auto intrinsic = opt_camera_trajectory.parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = opt_camera_trajectory.parameters_[c].extrinsic_;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector6d& J_r, double& r,
                                double& w) {
                w = 1.0;  // Dummy.
                ComputeJacobianAndResidualRigid(
                        i, J_r, r, w, opt_mesh, proxy_intensity, images_gray[c],
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
            opt_camera_trajectory.parameters_[c].extrinsic_ = pose;
#pragma omp critical
            {
                residual += r2;
                total_num_ += int(visibility_image_to_vertex[c].size());
            }
        }
        utility::LogDebug("Residual error : {:.6f} (avg : {:.6f})", residual,
                          residual / total_num_);
        SetProxyIntensityForVertex(opt_mesh, images_gray, utility::nullopt,
                                   opt_camera_trajectory,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);

        if (!option.debug_output_dir_.empty()) {
            // Save opt_mesh.
            SetGeometryColorAverage(opt_mesh, images_color, utility::nullopt,
                                    opt_camera_trajectory,
                                    visibility_vertex_to_image,
                                    option.image_boundary_margin_,
                                    option.invisible_vertex_color_knn_);
            std::string file_name = fmt::format(
                    "{}/iter_{}.ply",
                    option.debug_output_dir_ + "/rigid/opt_mesh", itr);
            io::WriteTriangleMesh(file_name, opt_mesh);

            // Save opt_camera_trajectory.
            file_name = fmt::format(
                    "{}/iter_{}.json",
                    option.debug_output_dir_ + "/rigid/opt_camera_trajectory",
                    itr);
            io::WritePinholeCameraTrajectory(file_name, opt_camera_trajectory);
        }
    }

    utility::LogDebug("[ColorMapOptimization] Set Mesh Color");
    SetGeometryColorAverage(opt_mesh, images_color, utility::nullopt,
                            opt_camera_trajectory, visibility_vertex_to_image,
                            option.image_boundary_margin_,
                            option.invisible_vertex_color_knn_);

    return opt_mesh;
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
