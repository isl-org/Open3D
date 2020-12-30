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

#include "open3d/pipelines/color_map/NonRigidOptimizer.h"

#include <memory>
#include <vector>

#include "open3d/io/ImageIO.h"
#include "open3d/io/ImageWarpingFieldIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/pipelines/color_map/ColorMapUtils.h"
#include "open3d/pipelines/color_map/ImageWarpingField.h"
#include "open3d/utility/FileSystem.h"

namespace Eigen {

typedef Eigen::Matrix<double, 14, 1> Vector14d;
typedef Eigen::Matrix<int, 14, 1> Vector14i;

}  // namespace Eigen

namespace open3d {
namespace pipelines {
namespace color_map {

static std::vector<ImageWarpingField> CreateWarpingFields(
        const std::vector<geometry::Image>& images,
        int number_of_vertical_anchors) {
    std::vector<ImageWarpingField> fields;
    for (size_t i = 0; i < images.size(); i++) {
        int width = images[i].width_;
        int height = images[i].height_;
        fields.push_back(
                ImageWarpingField(width, height, number_of_vertical_anchors));
    }
    return fields;
}

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: this function is almost identical to the functions in
/// Utility/Eigen.h/cpp, but this function takes additional multiplication
/// pattern that can produce JTJ having hundreds of rows and columns.
template <typename VecInTypeDouble,
          typename VecInTypeInt,
          typename MatOutType,
          typename VecOutType>
static std::tuple<MatOutType, VecOutType, double> ComputeJTJandJTrNonRigid(
        std::function<void(int, VecInTypeDouble&, double&, VecInTypeInt&)> f,
        int iteration_num,
        int nonrigidval,
        bool verbose /*=true*/) {
    MatOutType JTJ(6 + nonrigidval, 6 + nonrigidval);
    VecOutType JTr(6 + nonrigidval);
    double r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
#pragma omp parallel
    {
        MatOutType JTJ_private(6 + nonrigidval, 6 + nonrigidval);
        VecOutType JTr_private(6 + nonrigidval);
        double r2_sum_private = 0.0;
        JTJ_private.setZero();
        JTr_private.setZero();
        VecInTypeDouble J_r;
        VecInTypeInt pattern;
        double r;
#pragma omp for nowait
        for (int i = 0; i < iteration_num; i++) {
            f(i, J_r, r, pattern);
            for (auto x = 0; x < J_r.size(); x++) {
                for (auto y = 0; y < J_r.size(); y++) {
                    JTJ_private(pattern(x), pattern(y)) += J_r(x) * J_r(y);
                }
            }
            for (auto x = 0; x < J_r.size(); x++) {
                JTr_private(pattern(x)) += r * J_r(x);
            }
            r2_sum_private += r * r;
        }
#pragma omp critical
        {
            JTJ += JTJ_private;
            JTr += JTr_private;
            r2_sum += r2_sum_private;
        }
    }
    if (verbose) {
        utility::LogDebug("Residual : {:.2e} (# of elements : {:d})",
                          r2_sum / (double)iteration_num, iteration_num);
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), r2_sum);
}

static void ComputeJacobianAndResidualNonRigid(
        int row,
        Eigen::Vector14d& J_r,
        double& r,
        Eigen::Vector14i& pattern,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const geometry::Image& images_gray,
        const geometry::Image& images_dx,
        const geometry::Image& images_dy,
        const ImageWarpingField& warping_fields,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visibility_image_to_vertex,
        const int image_boundary_margin) {
    J_r.setZero();
    pattern.setZero();
    r = 0;
    int anchor_w = warping_fields.anchor_w_;
    double anchor_step = warping_fields.anchor_step_;
    int vid = visibility_image_to_vertex[row];
    Eigen::Vector3d V = mesh.vertices_[vid];
    Eigen::Vector4d G = extrinsic * Eigen::Vector4d(V(0), V(1), V(2), 1);
    Eigen::Vector4d L = intrinsic * G;
    double u = L(0) / L(2);
    double v = L(1) / L(2);
    if (!images_gray.TestImageBoundary(u, v, image_boundary_margin)) {
        return;
    }
    int ii = (int)(u / anchor_step);
    int jj = (int)(v / anchor_step);
    if (ii >= warping_fields.anchor_w_ - 1 ||
        jj >= warping_fields.anchor_h_ - 1) {
        return;
    }
    double p = (u - ii * anchor_step) / anchor_step;
    double q = (v - jj * anchor_step) / anchor_step;
    Eigen::Vector2d grids[4] = {
            warping_fields.QueryFlow(ii, jj),
            warping_fields.QueryFlow(ii, jj + 1),
            warping_fields.QueryFlow(ii + 1, jj),
            warping_fields.QueryFlow(ii + 1, jj + 1),
    };
    Eigen::Vector2d uuvv = (1 - p) * (1 - q) * grids[0] +
                           (1 - p) * (q)*grids[1] + (p) * (1 - q) * grids[2] +
                           (p) * (q)*grids[3];
    double uu = uuvv(0);
    double vv = uuvv(1);
    if (!images_gray.TestImageBoundary(uu, vv, image_boundary_margin)) {
        return;
    }
    bool valid;
    double gray, dIdfx, dIdfy;
    std::tie(valid, gray) = images_gray.FloatValueAt(uu, vv);
    std::tie(valid, dIdfx) = images_dx.FloatValueAt(uu, vv);
    std::tie(valid, dIdfy) = images_dy.FloatValueAt(uu, vv);
    Eigen::Vector2d dIdf(dIdfx, dIdfy);
    Eigen::Vector2d dfdx =
            ((grids[2] - grids[0]) * (1 - q) + (grids[3] - grids[1]) * q) /
            anchor_step;
    Eigen::Vector2d dfdy =
            ((grids[1] - grids[0]) * (1 - p) + (grids[3] - grids[2]) * p) /
            anchor_step;
    double dIdx = dIdf.dot(dfdx);
    double dIdy = dIdf.dot(dfdy);
    double invz = 1. / G(2);
    double v0 = dIdx * intrinsic(0, 0) * invz;
    double v1 = dIdy * intrinsic(1, 1) * invz;
    double v2 = -(v0 * G(0) + v1 * G(1)) * invz;
    J_r(0) = -G(2) * v1 + G(1) * v2;
    J_r(1) = G(2) * v0 - G(0) * v2;
    J_r(2) = -G(1) * v0 + G(0) * v1;
    J_r(3) = v0;
    J_r(4) = v1;
    J_r(5) = v2;
    J_r(6) = dIdf(0) * (1 - p) * (1 - q);
    J_r(7) = dIdf(1) * (1 - p) * (1 - q);
    J_r(8) = dIdf(0) * (1 - p) * (q);
    J_r(9) = dIdf(1) * (1 - p) * (q);
    J_r(10) = dIdf(0) * (p) * (1 - q);
    J_r(11) = dIdf(1) * (p) * (1 - q);
    J_r(12) = dIdf(0) * (p) * (q);
    J_r(13) = dIdf(1) * (p) * (q);
    pattern(0) = 0;
    pattern(1) = 1;
    pattern(2) = 2;
    pattern(3) = 3;
    pattern(4) = 4;
    pattern(5) = 5;
    pattern(6) = 6 + (ii + jj * anchor_w) * 2;
    pattern(7) = 6 + (ii + jj * anchor_w) * 2 + 1;
    pattern(8) = 6 + (ii + (jj + 1) * anchor_w) * 2;
    pattern(9) = 6 + (ii + (jj + 1) * anchor_w) * 2 + 1;
    pattern(10) = 6 + ((ii + 1) + jj * anchor_w) * 2;
    pattern(11) = 6 + ((ii + 1) + jj * anchor_w) * 2 + 1;
    pattern(12) = 6 + ((ii + 1) + (jj + 1) * anchor_w) * 2;
    pattern(13) = 6 + ((ii + 1) + (jj + 1) * anchor_w) * 2 + 1;
    r = (gray - proxy_intensity[vid]);
}

geometry::TriangleMesh RunNonRigidOptimizer(
        const geometry::TriangleMesh& mesh,
        const std::vector<geometry::RGBDImage>& images_rgbd,
        const camera::PinholeCameraTrajectory& camera_trajectory,
        const NonRigidOptimizerOption& option) {
    // The following properties will change during optimization.
    geometry::TriangleMesh opt_mesh = mesh;
    camera::PinholeCameraTrajectory opt_camera_trajectory = camera_trajectory;
    std::vector<ImageWarpingField> warping_fields;

    // The following properties remain unchanged during optimization.
    std::vector<geometry::Image> images_gray;
    std::vector<geometry::Image> images_dx;
    std::vector<geometry::Image> images_dy;
    std::vector<geometry::Image> images_color;
    std::vector<geometry::Image> images_depth;
    std::vector<geometry::Image> images_mask;
    std::vector<std::vector<int>> visibility_vertex_to_image;
    std::vector<std::vector<int>> visibility_image_to_vertex;
    std::vector<ImageWarpingField> warping_fields_init;

    // Create all debugging directories. We don't delete any existing files but
    // will overwrite them if the names are the same.
    if (!option.debug_output_dir_.empty()) {
        std::vector<std::string> dirs{
                option.debug_output_dir_,
                option.debug_output_dir_ + "/non_rigid",
                option.debug_output_dir_ + "/non_rigid/images_mask",
                option.debug_output_dir_ + "/non_rigid/opt_mesh",
                option.debug_output_dir_ + "/non_rigid/opt_camera_trajectory",
                option.debug_output_dir_ + "/non_rigid/warping_fields"};
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
                    option.debug_output_dir_ + "/non_rigid/images_mask", i);
            io::WriteImage(file_name, images_mask[i]);
        }
    }

    utility::LogDebug("[ColorMapOptimization] CreateVertexAndImageVisibility");
    std::tie(visibility_vertex_to_image, visibility_image_to_vertex) =
            CreateVertexAndImageVisibility(
                    opt_mesh, images_depth, images_mask, opt_camera_trajectory,
                    option.maximum_allowable_depth_,
                    option.depth_threshold_for_visibility_check_);

    utility::LogDebug("[ColorMapOptimization] Non-Rigid Optimization");
    warping_fields = CreateWarpingFields(images_gray,
                                         option.number_of_vertical_anchors_);
    warping_fields_init = CreateWarpingFields(
            images_gray, option.number_of_vertical_anchors_);
    std::vector<double> proxy_intensity;
    size_t n_vertex = opt_mesh.vertices_.size();
    int n_camera = int(opt_camera_trajectory.parameters_.size());
    SetProxyIntensityForVertex(opt_mesh, images_gray, warping_fields,
                               opt_camera_trajectory,
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
            pose = opt_camera_trajectory.parameters_[c].extrinsic_;

            auto intrinsic = opt_camera_trajectory.parameters_[c]
                                     .intrinsic_.intrinsic_matrix_;
            auto extrinsic = opt_camera_trajectory.parameters_[c].extrinsic_;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3, 3>(0, 0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&](int i, Eigen::Vector14d& J_r, double& r,
                                Eigen::Vector14i& pattern) {
                ComputeJacobianAndResidualNonRigid(
                        i, J_r, r, pattern, opt_mesh, proxy_intensity,
                        images_gray[c], images_dx[c], images_dy[c],
                        warping_fields[c], intr, extrinsic,
                        visibility_image_to_vertex[c],
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
            opt_camera_trajectory.parameters_[c].extrinsic_ = pose;

#pragma omp critical
            {
                residual += r2;
                residual_reg += rr_reg;
            }
        }
        utility::LogDebug("Residual error : {:.6f}, reg : {:.6f}", residual,
                          residual_reg);
        SetProxyIntensityForVertex(opt_mesh, images_gray, warping_fields,
                                   opt_camera_trajectory,
                                   visibility_vertex_to_image, proxy_intensity,
                                   option.image_boundary_margin_);

        if (!option.debug_output_dir_.empty()) {
            // Save opt_mesh.
            SetGeometryColorAverage(opt_mesh, images_color, warping_fields,
                                    opt_camera_trajectory,
                                    visibility_vertex_to_image,
                                    option.image_boundary_margin_,
                                    option.invisible_vertex_color_knn_);
            std::string file_name = fmt::format(
                    "{}/iter_{}.ply",
                    option.debug_output_dir_ + "/non_rigid/opt_mesh", itr);
            io::WriteTriangleMesh(file_name, opt_mesh);

            // Save opt_camera_trajectory.
            file_name = fmt::format("{}/iter_{}.json",
                                    option.debug_output_dir_ +
                                            "/non_rigid/opt_camera_trajectory",
                                    itr);
            io::WritePinholeCameraTrajectory(file_name, opt_camera_trajectory);

            // Save warping_fields.
            for (size_t i = 0; i < warping_fields.size(); ++i) {
                file_name = fmt::format(
                        "{}/iter_{}_camera_{}.json",
                        option.debug_output_dir_ + "/non_rigid/warping_fields",
                        itr, i);
                io::WriteImageWarpingField(file_name, warping_fields[i]);
            }
        }
    }

    utility::LogDebug("[ColorMapOptimization] Set Mesh Color");
    SetGeometryColorAverage(opt_mesh, images_color, warping_fields,
                            opt_camera_trajectory, visibility_vertex_to_image,
                            option.image_boundary_margin_,
                            option.invisible_vertex_color_knn_);

    return opt_mesh;
}

}  // namespace color_map
}  // namespace pipelines
}  // namespace open3d
