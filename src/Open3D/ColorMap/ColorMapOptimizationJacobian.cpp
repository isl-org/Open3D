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

#include "Open3D/ColorMap/ColorMapOptimizationJacobian.h"

#include "Open3D/ColorMap/EigenHelperForNonRigidOptimization.h"
#include "Open3D/ColorMap/ImageWarpingField.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
namespace color_map {
void ColorMapOptimizationJacobian::ComputeJacobianAndResidualRigid(
        int row,
        Eigen::Vector6d& J_r,
        double& r,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<geometry::Image>& images_gray,
        const std::shared_ptr<geometry::Image>& images_dx,
        const std::shared_ptr<geometry::Image>& images_dy,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visiblity_image_to_vertex,
        const int image_boundary_margin) {
    J_r.setZero();
    r = 0;
    int vid = visiblity_image_to_vertex[row];
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
}

void ColorMapOptimizationJacobian::ComputeJacobianAndResidualNonRigid(
        int row,
        Eigen::Vector14d& J_r,
        double& r,
        Eigen::Vector14i& pattern,
        const geometry::TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<geometry::Image>& images_gray,
        const std::shared_ptr<geometry::Image>& images_dx,
        const std::shared_ptr<geometry::Image>& images_dy,
        const ImageWarpingField& warping_fields,
        const ImageWarpingField& warping_fields_init,
        const Eigen::Matrix4d& intrinsic,
        const Eigen::Matrix4d& extrinsic,
        const std::vector<int>& visiblity_image_to_vertex,
        const int image_boundary_margin) {
    J_r.setZero();
    pattern.setZero();
    r = 0;
    int anchor_w = warping_fields.anchor_w_;
    double anchor_step = warping_fields.anchor_step_;
    int vid = visiblity_image_to_vertex[row];
    Eigen::Vector3d V = mesh.vertices_[vid];
    Eigen::Vector4d G = extrinsic * Eigen::Vector4d(V(0), V(1), V(2), 1);
    Eigen::Vector4d L = intrinsic * G;
    double u = L(0) / L(2);
    double v = L(1) / L(2);
    if (!images_gray->TestImageBoundary(u, v, image_boundary_margin)) {
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
    if (!images_gray->TestImageBoundary(uu, vv, image_boundary_margin)) {
        return;
    }
    bool valid;
    double gray, dIdfx, dIdfy;
    std::tie(valid, gray) = images_gray->FloatValueAt(uu, vv);
    std::tie(valid, dIdfx) = images_dx->FloatValueAt(uu, vv);
    std::tie(valid, dIdfy) = images_dy->FloatValueAt(uu, vv);
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
}  // namespace color_map
}  // namespace open3d
