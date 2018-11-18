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

#include "ColorMapOptimizationJacobian.h"

#include <Core/Geometry/Image.h>
#include <Core/Geometry/TriangleMesh.h>

namespace open3d {

void ColorMapOptimizationJacobian::ComputeJacobianAndResidualRigid(
        int row, std::vector<Eigen::Vector6d> &J_r, std::vector<double> &r,
        const TriangleMesh& mesh,
        const std::vector<double>& proxy_intensity,
        const std::shared_ptr<Image>& images_gray,
        const std::shared_ptr<Image>& images_dx,
        const std::shared_ptr<Image>& images_dy,
        const Eigen::Matrix4d &intrinsic,
        const Eigen::Matrix4d &extrinsic,
        const std::vector<int>& visiblity_image_to_vertex,
        const int image_boundary_margin)
{
    J_r.resize(1);
    r.resize(1);
    J_r[0].setZero();
    r[0] = 0;
    int vid = visiblity_image_to_vertex[row];
    Eigen::Vector3d x = mesh.vertices_[vid];
    Eigen::Vector4d g = extrinsic * Eigen::Vector4d(x(0), x(1), x(2), 1);
    Eigen::Vector4d uv = intrinsic * g;
    double u = uv(0) / uv(2);
    double v = uv(1) / uv(2);
    if (!images_gray->TestImageBoundary(u, v, image_boundary_margin))
        return;
    bool valid; double gray, dIdx, dIdy;
    std::tie(valid, gray) = images_gray->FloatValueAt(u, v);
    std::tie(valid, dIdx) = images_dx->FloatValueAt(u, v);
    std::tie(valid, dIdy) = images_dy->FloatValueAt(u, v);
    if (gray == -1.0)
        return;
    double invz = 1. / g(2);
    double v0 = dIdx * intrinsic(0, 0) * invz;
    double v1 = dIdy * intrinsic(1, 1) * invz;
    double v2 = -(v0 * g(0) + v1 * g(1)) * invz;
    J_r[0](0) = (-g(2) * v1 + g(1) * v2);
    J_r[0](1) = (g(2) * v0 - g(0) * v2);
    J_r[0](2) = (-g(1) * v0 + g(0) * v1);
    J_r[0](3) = v0;
    J_r[0](4) = v1;
    J_r[0](5) = v2;
    r[0] = (gray - proxy_intensity[vid]);
}

}    // namespace open3d
