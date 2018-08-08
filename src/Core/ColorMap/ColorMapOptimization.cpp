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

#include <Eigen/Dense>

#include <Core/Utility/Console.h>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Core/Camera/PinholeCameraTrajectory.h>
#include <IO/ClassIO/TriangleMeshIO.h>
#include <Core/Utility/Eigen.h>

namespace open3d {

namespace {

class ImageWarpingField {

public:
    ImageWarpingField (int width, int height, int number_of_vertical_anchors) {
        InitializeWarpingFields(width, height, number_of_vertical_anchors);
    }

    void InitializeWarpingFields(int width, int height,
            int number_of_vertical_anchors) {
        anchor_h_ = number_of_vertical_anchors;
        anchor_step_ = double(height) / (anchor_h_ - 1);
        anchor_w_ = int(std::ceil(double(width) / anchor_step_) + 1);
        flow_ = Eigen::VectorXd(anchor_w_ * anchor_h_ * 2);
        for (int i = 0; i <= (anchor_w_ - 1); i++) {
            for (int j = 0; j <= (anchor_h_ - 1); j++) {
                int baseidx = (i + j * anchor_w_) * 2;
                flow_(baseidx) = i * anchor_step_;
                flow_(baseidx + 1) = j * anchor_step_;
            }
        }
    }

    Eigen::Vector2d QueryFlow(int i, int j) const {
        int baseidx = (i + j * anchor_w_) * 2;
        // exceptional case: quried anchor index is out of pre-defined space
        if (baseidx < 0 || baseidx > anchor_w_ * anchor_h_ * 2)
            return Eigen::Vector2d(0.0, 0.0);
        else
            return Eigen::Vector2d(flow_(baseidx), flow_(baseidx + 1));
    }

    Eigen::Vector2d GetImageWarpingField(double u, double v) const {
        int i = (int)(u / anchor_step_);
        int j = (int)(v / anchor_step_);
        double p = (u - i * anchor_step_) / anchor_step_;
        double q = (v - j * anchor_step_) / anchor_step_;
        return (1 - p) * (1 - q) * QueryFlow(i, j)
            + (1 - p) * (q)* QueryFlow(i, j + 1)
            + (p)* (1 - q) * QueryFlow(i + 1, j)
            + (p)* (q)* QueryFlow(i + 1, j + 1);
    }

public:
    Eigen::VectorXd flow_;
    int anchor_w_, anchor_h_;
    double anchor_step_;
};

const double IMAGE_BOUNDARY_MARGIN = 10;

inline std::tuple<float, float, float> Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const PinholeCameraTrajectory& camera, int camid)
{
    std::pair<double, double> f = camera.intrinsic_.GetFocalLength();
    std::pair<double, double> p = camera.intrinsic_.GetPrincipalPoint();
    Eigen::Vector4d Vt = camera.extrinsic_[camid] *
            Eigen::Vector4d(X(0), X(1), X(2), 1);
    float u = float((Vt(0) * f.first) / Vt(2) + p.first);
    float v = float((Vt(1) * f.second) / Vt(2) + p.second);
    float z = float(Vt(2));
    return std::make_tuple(u, v, z);
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
        MakeVertexAndImageVisibility(const TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        const std::vector<Image>& images_mask,
        const PinholeCameraTrajectory& camera,
        const ColorMapOptmizationOption& option)
{
    auto n_camera = camera.extrinsic_.size();
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
            if (*PointerAt<unsigned char>(images_mask[c], u_d, v_d) == 255)
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
        const ColorMapOptmizationOption& option)
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
        const ColorMapOptmizationOption& option)
{
    auto n_vertex = mesh.vertices_.size();
    auto n_camera = camera.extrinsic_.size();
    Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
    intr.block<3,3>(0,0) = camera.intrinsic_.intrinsic_matrix_;
    intr(3, 3) = 1.0;
    double fx = intr(0, 0);
    double fy = intr(1, 1);
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
        for (int i = 0; i < n_camera; i++) {
            int nonrigidval = warping_fields[i].anchor_w_ *
                    warping_fields[i].anchor_h_ * 2;
            Eigen::MatrixXd JJ = Eigen::MatrixXd::Zero(
                    6 + nonrigidval, 6 + nonrigidval);
            Eigen::VectorXd Jb = Eigen::VectorXd::Zero(6 + nonrigidval);
            double rr = 0.0;
            double rr_reg = 0.0;
            int this_num = 0;
            Eigen::Matrix4d pose;
            pose = camera.extrinsic_[i];
            double anchor_step = warping_fields[i].anchor_step_;
            int anchor_w = warping_fields[i].anchor_w_;
            for (auto iter = 0; iter < visiblity_image_to_vertex[i].size();
                    iter++) {
                int j = visiblity_image_to_vertex[i][iter];
                Eigen::Vector3d V = mesh.vertices_[j];
                Eigen::Vector4d G = pose * Eigen::Vector4d(V(0), V(1), V(2), 1);
                Eigen::Vector4d L = intr * G;
                double u = L(0) / L(2);
                double v = L(1) / L(2);
                int ii = (int)(u / anchor_step);
                int jj = (int)(v / anchor_step);
                double p = (u - ii * anchor_step) / anchor_step;
                double q = (v - jj * anchor_step) / anchor_step;
                Eigen::Vector2d grids[4] = {
                    warping_fields[i].QueryFlow(ii, jj),
                    warping_fields[i].QueryFlow(ii, jj + 1),
                    warping_fields[i].QueryFlow(ii + 1, jj),
                    warping_fields[i].QueryFlow(ii + 1, jj + 1),
                };
                Eigen::Vector2d uuvv = (1 - p) * (1 - q) * grids[0]
                    + (1 - p) * (q)* grids[1]
                    + (p)* (1 - q) * grids[2]
                    + (p)* (q)* grids[3];
                double uu = uuvv(0);
                double vv = uuvv(1);
                if (!images_gray[i]->TestImageBoundary(uu, vv,
                        IMAGE_BOUNDARY_MARGIN))
                    continue;
                bool valid; double gray, dIdfx, dIdfy;
                std::tie(valid, gray) = images_gray[i]->FloatValueAt(uu, vv);
                std::tie(valid, dIdfx) = images_dx[i]->FloatValueAt(uu, vv);
                std::tie(valid, dIdfy) = images_dy[i]->FloatValueAt(uu, vv);
                Eigen::Vector2d dIdf(dIdfx, dIdfy);
                Eigen::Vector2d dfdx = ((grids[2] - grids[0]) * (1 - q) +
                        (grids[3] - grids[1]) * q) / anchor_step;
                Eigen::Vector2d dfdy = ((grids[1] - grids[0]) * (1 - p) +
                        (grids[3] - grids[2]) * p) / anchor_step;
                double dIdx = dIdf.dot(dfdx);
                double dIdy = dIdf.dot(dfdy);
                double invz = 1. / G(2);
                double v0 = dIdx * fx * invz;
                double v1 = dIdy * fy * invz;
                double v2 = -(v0 * G(0) + v1 * G(1)) * invz;
                double C[6 + 8];
                C[0] = -G(2) * v1 + G(1) * v2;
                C[1] = G(2) * v0 - G(0) * v2;
                C[2] = -G(1) * v0 + G(0) * v1;
                C[3] = v0;
                C[4] = v1;
                C[5] = v2;
                C[6] = dIdf(0) * (1 - p) * (1 - q);
                C[7] = dIdf(1) * (1 - p) * (1 - q);
                C[8] = dIdf(0) * (1 - p) * (q);
                C[9] = dIdf(1) * (1 - p) * (q);
                C[10] = dIdf(0) * (p)* (1 - q);
                C[11] = dIdf(1) * (p)* (1 - q);
                C[12] = dIdf(0) * (p)* (q);
                C[13] = dIdf(1) * (p)* (q);
                int idx[6 + 8];
                idx[0] = 0;
                idx[1] = 1;
                idx[2] = 2;
                idx[3] = 3;
                idx[4] = 4;
                idx[5] = 5;
                idx[6] = 6 + (ii + jj * anchor_w) * 2;
                idx[7] = 6 + (ii + jj * anchor_w) * 2 + 1;
                idx[8] = 6 + (ii + (jj + 1) * anchor_w) * 2;
                idx[9] = 6 + (ii + (jj + 1) * anchor_w) * 2 + 1;
                idx[10] = 6 + ((ii + 1) + jj * anchor_w) * 2;
                idx[11] = 6 + ((ii + 1) + jj * anchor_w) * 2 + 1;
                idx[12] = 6 + ((ii + 1) + (jj + 1) * anchor_w) * 2;
                idx[13] = 6 + ((ii + 1) + (jj + 1) * anchor_w) * 2 + 1;
                for (int x = 0; x < 14; x++) {
                    for (int y = 0; y < 14; y++) {
                        JJ(idx[x], idx[y]) += C[x] * C[y];
                    }
                }
                double r = (proxy_intensity[j] - gray);
                for (int x = 0; x < 14; x++) {
                    Jb(idx[x]) -= r * C[x];
                }
                rr += r * r;
                this_num++;
            }
            if (this_num == 0)
                continue;
            double weight = option.non_rigid_anchor_point_weight_
                    * this_num / n_vertex;
            for (int j = 0; j < nonrigidval; j++) {
                double r = weight * (warping_fields[i].flow_(j) -
                        warping_fields_init[i].flow_(j));
                JJ(6 + j, 6 + j) += weight * weight;
                Jb(6 + j) += weight * r;
                rr_reg += r * r;
            }
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                bool success = false;
                Eigen::VectorXd result;
                std::tie(success, result) = SolveLinearSystem(JJ, -Jb, false);
                Eigen::Affine3d aff_mat;
                aff_mat.linear() = (Eigen::Matrix3d)
                        Eigen::AngleAxisd(result(2),Eigen::Vector3d::UnitZ())
                        * Eigen::AngleAxisd(result(1),Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(result(0),Eigen::Vector3d::UnitX());
                aff_mat.translation() =
                        Eigen::Vector3d(result(3), result(4), result(5));
                pose = aff_mat.matrix() * pose;
                for (int j = 0; j < nonrigidval; j++) {
                    warping_fields[i].flow_(j) += result(6 + j);
                }
            }
            camera.extrinsic_[i] = pose;
            residual += rr;
            residual_reg += rr_reg;
            total_num_ += this_num;
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
        const ColorMapOptmizationOption& option)
{
    int total_num_ = 0;
    auto n_camera = camera.extrinsic_.size();
    Eigen::Matrix4d intr = Eigen::Matrix4d::Zero();
    intr.block<3,3>(0,0) = camera.intrinsic_.intrinsic_matrix_;
    intr(3, 3) = 1.0;
    double fx = intr(0, 0);
    double fy = intr(1, 1);
    SetProxyIntensityForVertex(mesh, images_gray, camera,
            visiblity_vertex_to_image, proxy_intensity);
    for (int itr = 0; itr < option.maximum_iteration_; itr++) {
        PrintDebug("[Iteration %04d] ", itr+1);
        double residual = 0.0;
        total_num_ = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_camera; i++) {
            Eigen::MatrixXd JJ(6, 6);
            Eigen::VectorXd Jb(6);
            JJ.setZero();
            Jb.setZero();
            double rr = 0.0;
            int this_num = 0;
            Eigen::Matrix4d pose;
            pose = camera.extrinsic_[i];
            for (auto iter = 0; iter < visiblity_image_to_vertex[i].size();
                    iter++) {
                int j = visiblity_image_to_vertex[i][iter];
                Eigen::Vector3d V = mesh.vertices_[j];
                Eigen::Vector4d G = pose * Eigen::Vector4d(V(0), V(1), V(2), 1);
                Eigen::Vector4d L = intr * G;
                double u = L(0) / L(2);
                double v = L(1) / L(2);
                if (!images_gray[i]->TestImageBoundary(u, v,
                        IMAGE_BOUNDARY_MARGIN))
                    continue;
                bool valid; double gray, dIdx, dIdy;
                std::tie(valid, gray) = images_gray[i]->FloatValueAt(u, v);
                std::tie(valid, dIdx) = images_dx[i]->FloatValueAt(u, v);
                std::tie(valid, dIdy) = images_dy[i]->FloatValueAt(u, v);
                if (gray == -1.0)
                    continue;
                double invz = 1. / G(2);
                double v0 = dIdx * fx * invz;
                double v1 = dIdy * fy * invz;
                double v2 = -(v0 * G(0) + v1 * G(1)) * invz;
                double C[6];
                C[0] = (-G(2) * v1 + G(1) * v2);
                C[1] = (G(2) * v0 - G(0) * v2);
                C[2] = (-G(1) * v0 + G(0) * v1);
                C[3] = v0;
                C[4] = v1;
                C[5] = v2;
                for (int x = 0; x < 6; x++) {
                    for (int y = 0; y < 6; y++) {
                        JJ(x, y) += C[x] * C[y];
                    }
                }
                double r = (proxy_intensity[j] - gray);
                for (int x = 0; x < 6; x++) {
                    Jb(x) -= r * C[x];
                }
                rr += r * r;
                this_num++;
            }
            Eigen::VectorXd result(6);
            result = -JJ.inverse() * Jb;
            Eigen::Affine3d aff_mat;
            aff_mat.linear() = (Eigen::Matrix3d)
                    Eigen::AngleAxisd(result(2), Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
            aff_mat.translation() =
                    Eigen::Vector3d(result(3), result(4), result(5));
            pose = aff_mat.matrix() * pose;
            camera.extrinsic_[i] = pose;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                residual += rr;
                total_num_ += this_num;
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

std::vector<Image> MakeDepthMasks(
        const std::vector<RGBDImage>& images_rgbd,
        const ColorMapOptmizationOption& option)
{
    auto n_images = images_rgbd.size();
    std::vector<Image> images_mask;
    for (int i=0; i<n_images; i++) {
        PrintDebug("[MakeDepthMasks] Image %d/%d\n", i, n_images);
        int width = images_rgbd[i].depth_.width_;
        int height = images_rgbd[i].depth_.height_;
        auto depth_image = CreateFloatImageFromImage(images_rgbd[i].depth_);
        auto depth_image_gradient_dx = FilterImage(*depth_image,
                Image::FilterType::Sobel3Dx);
        auto depth_image_gradient_dy = FilterImage(*depth_image,
                Image::FilterType::Sobel3Dy);
        auto mask = std::make_shared<Image>();
        mask->PrepareImage(width, height, 1, 1);
        for (int v=0; v<height; v++) {
            for (int u=0; u<width; u++) {
                double dx = *PointerAt<float>(*depth_image_gradient_dx, u, v);
                double dy = *PointerAt<float>(*depth_image_gradient_dy, u, v);
                double mag = sqrt(dx * dx + dy * dy);
                if (mag > option.depth_threshold_for_discontinuity_check_) {
                    *PointerAt<unsigned char>(*mask, u, v) = 255;
                } else {
                    *PointerAt<unsigned char>(*mask, u, v) = 0;
                }
            }
        }
        auto mask_dilated = DilateImage(*mask,
                option.half_dilation_kernel_size_for_discontinuity_map_);
        images_mask.push_back(*mask_dilated);
    }
    return std::move(images_mask);
}

}    // unnamed namespace

void ColorMapOptimization(TriangleMesh& mesh,
        const std::vector<RGBDImage>& images_rgbd,
        PinholeCameraTrajectory& camera,
        const ColorMapOptmizationOption& option
        /* = ColorMapOptmizationOption()*/)
{
    PrintDebug("[ColorMapOptimization]\n");
    std::vector<std::shared_ptr<Image>> images_gray;
    std::vector<std::shared_ptr<Image>> images_dx;
    std::vector<std::shared_ptr<Image>> images_dy;
    std::tie(images_gray, images_dx, images_dy) =
            MakeGradientImages(images_rgbd);

    PrintDebug("[ColorMapOptimization] :: MakingMasks\n");
    auto images_mask = MakeDepthMasks(images_rgbd, option);

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
