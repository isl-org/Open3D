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

#include "ColorMapOptimization.h"

#include <Core/Camera/PinholeCameraTrajectory.h>
#include <Core/ColorMap/ColorMapOptimizationJacobian.h>
#include <Core/ColorMap/ImageWarpingField.h>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Eigen.h>

#include <iostream>

namespace open3d {

namespace {

const double IMAGE_BOUNDARY_MARGIN = 10;

inline std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const PinholeCameraTrajectory& camera, int camid)
{
    std::pair<double, double> f =
            camera.parameters_[camid].intrinsic_.GetFocalLength();
    std::pair<double, double> p =
            camera.parameters_[camid].intrinsic_.GetPrincipalPoint();
    Eigen::Vector4d Vt = camera.parameters_[camid].extrinsic_ *
            Eigen::Vector4d(X(0), X(1), X(2), 1);
    float u = float((Vt(0) * f.first) / Vt(2) + p.first);
    float v = float((Vt(1) * f.second) / Vt(2) + p.second);
    float z = float(Vt(2));
    return std::make_tuple(u, v, z);
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
        MakeVertexAndImageVisibility(const TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const std::vector<std::shared_ptr<Image>>& images_mask,
        const PinholeCameraTrajectory& camera,
        const ColorMapOptimizationOption& option)
{
    auto n_camera = camera.parameters_.size();
    auto n_vertex = mesh.vertices_.size();
    std::vector<std::vector<int>> visiblity_vertex_to_image;
    std::vector<std::vector<int>> visiblity_image_to_vertex;
    visiblity_vertex_to_image.resize(n_vertex);
    visiblity_image_to_vertex.resize(n_camera);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < n_camera; c++) {
        int viscnt = 0;
        for (int vertex_id = 0; vertex_id < n_vertex; vertex_id++) {
            Eigen::Vector3d X = mesh.vertices_[vertex_id];
            float u, v, d;
            std::tie(u, v, d) = Project3DPointAndGetUVDepth(X, camera, c);
            int u_d = int(round(u)), v_d = int(round(v));
            if (d < 0.0 || !images_rgbd[c].depth_.TestImageBoundary(u_d, v_d))
                continue;
            float d_sensor = *PointerAt<float>(images_rgbd[c].depth_, u_d, v_d);
            if (d_sensor > option.maximum_allowable_depth_)
                continue;
            if (*PointerAt<unsigned char>(*images_mask[c], u_d, v_d) == 255)
                continue;
            if (std::fabs(d - d_sensor) <
                    option.depth_threshold_for_visiblity_check_) {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    visiblity_vertex_to_image[vertex_id].push_back(c);
                    visiblity_image_to_vertex[c].push_back(vertex_id);
                    viscnt++;
                }
            }
        }
        PrintDebug("[cam %d] %.5f percents are visible\n",
                c, double(viscnt) / n_vertex * 100); fflush(stdout);
    }
    return std::move(std::make_tuple(
            visiblity_vertex_to_image, visiblity_image_to_vertex));
}

std::vector<ImageWarpingField> MakeWarpingFields(
        const std::vector<std::shared_ptr<Image>>& images,
        const ColorMapOptimizationOption& option)
{
    std::vector<ImageWarpingField> fields;
    for (auto i = 0; i < images.size(); i++) {
        int width = images[i]->width_;
        int height = images[i]->height_;
        fields.push_back(ImageWarpingField(width, height,
                option.number_of_vertical_anchors_));
    }
    return std::move(fields);
}

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid, int ch = -1)
{
    float u, v, depth;
    std::tie(u, v, depth) = Project3DPointAndGetUVDepth(V, camera, camid);
    if (img.TestImageBoundary(u, v, IMAGE_BOUNDARY_MARGIN)) {
        int u_round = int(round(u));
        int v_round = int(round(v));
        if (ch == -1) {
            return std::make_tuple(true,
                    *PointerAt<T>(img, u_round, v_round));
        } else {
            return std::make_tuple(true,
                    *PointerAt<T>(img, u_round, v_round, ch));
        }
    } else {
        return std::make_tuple(false, 0);
    }
}

template<typename T>
std::tuple<bool, T> QueryImageIntensity(
        const Image& img, const ImageWarpingField& field,
        const Eigen::Vector3d& V,
        const PinholeCameraTrajectory& camera, int camid, int ch = -1)
{
    float u, v, depth;
    std::tie(u, v, depth) = Project3DPointAndGetUVDepth(V, camera, camid);
    if (img.TestImageBoundary(u, v, IMAGE_BOUNDARY_MARGIN)) {
        Eigen::Vector2d uv_shift = field.GetImageWarpingField(u, v);
        if (img.TestImageBoundary(uv_shift(0), uv_shift(1),
                IMAGE_BOUNDARY_MARGIN)) {
            int u_shift = int(round(uv_shift(0)));
            int v_shift = int(round(uv_shift(1)));
            if (ch == -1) {
                return std::make_tuple(true,
                        *PointerAt<T>(img, u_shift, v_shift));
            } else {
                return std::make_tuple(true,
                        *PointerAt<T>(img, u_shift, v_shift, ch));
            }
        }
    }
    return std::make_tuple(false, 0);
}

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<ImageWarpingField>& warping_field,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity)
{
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#pragma omp parallel for schedule(static)
    for (auto i = 0; i < n_vertex; i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (auto iter = 0; iter < visiblity_vertex_to_image[i].size();
                iter++) {
            int j = visiblity_vertex_to_image[i][iter];
            float gray;
            bool valid = false;
            std::tie(valid, gray) = QueryImageIntensity<float>(
                    *images_gray[j], warping_field[j],
                    mesh.vertices_[i], camera, j);
            if (valid) {
                sum += 1.0;
                proxy_intensity[i] += gray;
            }
        }
        if (sum > 0) {
            proxy_intensity[i] /= sum;
        }
    }
}

void SetProxyIntensityForVertex(const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        std::vector<double>& proxy_intensity)
{
    auto n_vertex = mesh.vertices_.size();
    proxy_intensity.resize(n_vertex);

#pragma omp parallel for num_threads( 8 )
    for (auto i = 0; i < n_vertex; i++) {
        proxy_intensity[i] = 0.0;
        float sum = 0.0;
        for (auto iter = 0; iter < visiblity_vertex_to_image[i].size();
                iter++) {
            int j = visiblity_vertex_to_image[i][iter];
            float gray;
            bool valid = false;
            std::tie(valid, gray) = QueryImageIntensity<float>(
                    *images_gray[j], mesh.vertices_[i], camera, j);
            if (valid) {
                sum += 1.0;
                proxy_intensity[i] += gray;
            }
        }
        if (sum > 0) {
            proxy_intensity[i] /= sum;
        }
    }
}

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
        const ColorMapOptimizationOption& option)
{
    auto n_vertex = mesh.vertices_.size();
    auto n_camera = camera.parameters_.size();
    SetProxyIntensityForVertex(mesh, images_gray, warping_fields, camera,
            visiblity_vertex_to_image, proxy_intensity);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        PrintDebug("[Iteration %04d] ", itr+1);
        double residual = 0.0;
        double residual_reg = 0.0;
        int total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
            intr.block<3,3>(0,0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&]
            (int i, Eigen::Vector14d &J_r, double &r, Eigen::Vector14d &pattern) {
                jac.ComputeJacobianAndResidualNonRigid(i, J_r, r, pattern,
                        mesh, proxy_intensity,
                        images_gray[c], images_dx[c], images_dy[c],
                        warping_fields[c], warping_fields_init[c],
                        intr, extrinsic, visiblity_image_to_vertex[c],
                        IMAGE_BOUNDARY_MARGIN);
            };
            Eigen::MatrixXd JTJ;
            Eigen::VectorXd JTr;
            double r2;
            int this_num = visiblity_image_to_vertex[c].size();
            std::tie(JTJ, JTr, r2) = ComputeJTJandJTr
            <Eigen::Vector14d, Eigen::MatrixXd, Eigen::VectorXd>(f_lambda,
                    visiblity_image_to_vertex[c].size(), nonrigidval, false);

            double weight = option.non_rigid_anchor_point_weight_
                    * this_num / n_vertex;
            for (int j = 0; j < nonrigidval; j++) {
                double r = weight * (warping_fields[c].flow_(j) -
                        warping_fields_init[c].flow_(j));
                JTJ(6 + j, 6 + j) += weight * weight;
                JTr(6 + j) += weight * r;
                rr_reg += r * r;
            }

            bool success;
            Eigen::VectorXd result;
            std::tie(success, result) = SolveLinearSystem(JTJ, -JTr, false);
            Eigen::Vector6d result_pose;
            result_pose << result.block(0,0,6,1);
            auto delta = TransformVector6dToMatrix4d(result_pose);
            pose = delta * pose;

            for (int j = 0; j < nonrigidval; j++) {
                warping_fields[c].flow_(j) += result(6 + j);
            }
            camera.parameters_[c].extrinsic_ = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                residual_reg += rr_reg;
                total_num_ += this_num;
            }
        }
        PrintDebug("Residual error : %.6f, reg : %.6f\n",
                residual, residual_reg);
        SetProxyIntensityForVertex(mesh, images_gray, warping_fields, camera,
                visiblity_vertex_to_image, proxy_intensity);
    }
}

void OptimizeImageCoorRigid(
        const TriangleMesh& mesh,
        const std::vector<std::shared_ptr<Image>>& images_gray,
        const std::vector<std::shared_ptr<Image>>& images_dx,
        const std::vector<std::shared_ptr<Image>>& images_dy,
        PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image,
        const std::vector<std::vector<int>>& visiblity_image_to_vertex,
        std::vector<double>& proxy_intensity,
        const ColorMapOptimizationOption& option)
{
    int total_num_ = 0;
    auto n_camera = camera.parameters_.size();
    SetProxyIntensityForVertex(mesh, images_gray, camera,
            visiblity_vertex_to_image, proxy_intensity);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        PrintDebug("[Iteration %04d] ", itr+1);
        double residual = 0.0;
        total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < n_camera; c++) {
            Eigen::Matrix4d pose;
            pose = camera.parameters_[c].extrinsic_;

            auto intrinsic = camera.parameters_[c].intrinsic_.intrinsic_matrix_;
            auto extrinsic = camera.parameters_[c].extrinsic_;
            ColorMapOptimizationJacobian jac;
            Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
            intr.block<3,3>(0,0) = intrinsic;
            intr(3, 3) = 1.0;

            auto f_lambda = [&]
            (int i, Eigen::Vector6d &J_r, double &r) {
                    jac.ComputeJacobianAndResidualRigid(i, J_r, r,
                    mesh, proxy_intensity,
                    images_gray[c], images_dx[c], images_dy[c],
                    intr, extrinsic, visiblity_image_to_vertex[c],
                    IMAGE_BOUNDARY_MARGIN);
            };
            Eigen::Matrix6d JTJ;
            Eigen::Vector6d JTr;
            double r2;
            std::tie(JTJ, JTr, r2) = ComputeJTJandJTr
                    <Eigen::Matrix6d, Eigen::Vector6d>(f_lambda,
                    visiblity_image_to_vertex[c].size(), false);

            bool is_success;
            Eigen::Matrix4d delta;
            std::tie(is_success, delta) =
                    SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
            pose = delta * pose;
            camera.parameters_[c].extrinsic_ = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += r2;
                total_num_ += visiblity_image_to_vertex[c].size();
            }
        }
        PrintDebug("Residual error : %.6f (avg : %.6f)\n",
                residual, residual / total_num_);
        SetProxyIntensityForVertex(mesh, images_gray, camera,
                visiblity_vertex_to_image, proxy_intensity);
    }
}

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const std::vector<ImageWarpingField>& warping_fields,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image)
{
    auto n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (auto iter = 0; iter < visiblity_vertex_to_image[i].size();
                iter++) {
            int j = visiblity_vertex_to_image[i][iter];
            unsigned char r_temp, g_temp, b_temp;
            bool valid = false;
            std::tie(valid, r_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, warping_fields[j],
                    mesh.vertices_[i], camera, j, 0);
            std::tie(valid, g_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, warping_fields[j],
                    mesh.vertices_[i], camera, j, 1);
            std::tie(valid, b_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, warping_fields[j],
                    mesh.vertices_[i], camera, j, 2);
            float r = (float)r_temp / 255.0f;
            float g = (float)g_temp / 255.0f;
            float b = (float)b_temp / 255.0f;
            if (valid) {
                mesh.vertex_colors_[i] += Eigen::Vector3d(r, g, b);
                sum += 1.0;
            }
        }
        if (sum > 0.0) {
            mesh.vertex_colors_[i] /= sum;
        }
    }
}

void SetGeometryColorAverage(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const PinholeCameraTrajectory& camera,
        const std::vector<std::vector<int>>& visiblity_vertex_to_image)
{
    auto n_vertex = mesh.vertices_.size();
    mesh.vertex_colors_.clear();
    mesh.vertex_colors_.resize(n_vertex);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_vertex; i++) {
        mesh.vertex_colors_[i] = Eigen::Vector3d::Zero();
        double sum = 0.0;
        for (auto iter = 0; iter < visiblity_vertex_to_image[i].size();
                iter++) {
            int j = visiblity_vertex_to_image[i][iter];
            unsigned char r_temp, g_temp, b_temp;
            bool valid = false;
            std::tie(valid, r_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, mesh.vertices_[i], camera, j, 0);
            std::tie(valid, g_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, mesh.vertices_[i], camera, j, 1);
            std::tie(valid, b_temp) = QueryImageIntensity<unsigned char>(
                    images_rgbd[j].color_, mesh.vertices_[i], camera, j, 2);
            float r = (float)r_temp / 255.0f;
            float g = (float)g_temp / 255.0f;
            float b = (float)b_temp / 255.0f;
            if (valid) {
                mesh.vertex_colors_[i] += Eigen::Vector3d(r, g, b);
                sum += 1.0;
            }
        }
        if (sum > 0.0) {
            mesh.vertex_colors_[i] /= sum;
        }
    }
}

std::tuple<std::vector<std::shared_ptr<Image>>,
        std::vector<std::shared_ptr<Image>>,
        std::vector<std::shared_ptr<Image>>> MakeGradientImages(
        const std::vector<RGBDImage>& images_rgbd)
{
    auto n_images = images_rgbd.size();
    std::vector<std::shared_ptr<Image>> images_gray;
    std::vector<std::shared_ptr<Image>> images_dx;
    std::vector<std::shared_ptr<Image>> images_dy;
    for (int i=0; i<n_images; i++) {
        auto gray_image = CreateFloatImageFromImage(images_rgbd[i].color_);
        auto gray_image_filtered = FilterImage(*gray_image,
                Image::FilterType::Gaussian3);
        images_gray.push_back(gray_image_filtered);
        images_dx.push_back(FilterImage(*gray_image_filtered,
                Image::FilterType::Sobel3Dx));
        images_dy.push_back(FilterImage(*gray_image_filtered,
                Image::FilterType::Sobel3Dy));
    }
    return std::move(std::make_tuple(images_gray, images_dx, images_dy));
}

std::vector<std::shared_ptr<Image>> CreateDepthBoundaryMasks(
        const std::vector<RGBDImage>& images_rgbd,
        const ColorMapOptimizationOption& option)
{
    auto n_images = images_rgbd.size();
    std::vector<std::shared_ptr<Image>> masks;
    for (auto i=0; i<n_images; i++) {
        PrintDebug("[MakeDepthMasks] Image %d/%d\n", i, n_images);
        masks.push_back(CreateDepthBoundaryMask(images_rgbd[i].depth_,
                option.depth_threshold_for_discontinuity_check_,
                option.half_dilation_kernel_size_for_discontinuity_map_));
    }
    return masks;
}

}    // unnamed namespace

void ColorMapOptimization(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        PinholeCameraTrajectory& camera,
        const ColorMapOptimizationOption& option
        /* = ColorMapOptimizationOption()*/)
{
    PrintDebug("[ColorMapOptimization]\n");
    std::vector<std::shared_ptr<Image>> images_gray;
    std::vector<std::shared_ptr<Image>> images_dx;
    std::vector<std::shared_ptr<Image>> images_dy;
    std::tie(images_gray, images_dx, images_dy) =
            MakeGradientImages(images_rgbd);

    PrintDebug("[ColorMapOptimization] :: MakingMasks\n");
    auto images_mask = CreateDepthBoundaryMasks(images_rgbd, option);

    PrintDebug("[ColorMapOptimization] :: VisibilityCheck\n");
    std::vector<std::vector<int>> visiblity_vertex_to_image;
    std::vector<std::vector<int>> visiblity_image_to_vertex;
    std::tie(visiblity_vertex_to_image, visiblity_image_to_vertex) =
            MakeVertexAndImageVisibility(mesh, images_rgbd,
            images_mask, camera, option);

    std::vector<double> proxy_intensity;
    if (option.non_rigid_camera_coordinate_) {
        PrintDebug("[ColorMapOptimization] :: Non-Rigid Optimization\n");
        auto warping_uv_ = MakeWarpingFields(images_gray, option);
        auto warping_uv_init_ = MakeWarpingFields(images_gray, option);
        OptimizeImageCoorNonrigid(mesh, images_gray,
                images_dx, images_dy, warping_uv_, warping_uv_init_, camera,
                visiblity_vertex_to_image, visiblity_image_to_vertex,
                proxy_intensity, option);
        SetGeometryColorAverage(mesh, images_rgbd, warping_uv_, camera,
                visiblity_vertex_to_image);
    } else {
        PrintDebug("[ColorMapOptimization] :: Rigid Optimization\n");
        OptimizeImageCoorRigid(mesh, images_gray, images_dx, images_dy, camera,
                visiblity_vertex_to_image, visiblity_image_to_vertex,
                proxy_intensity, option);
        SetGeometryColorAverage(mesh, images_rgbd, camera,
                visiblity_vertex_to_image);
    }
}

}    // namespace open3d
