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

#include "open3d/t/pipelines/registration/TransformationEstimation.h"

#include <Eigen/Geometry>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // if (corres.empty()) return 0.0;
    double err = 0.0;
    // Vectorised
    //  source, target -> shared memory
    //  error_ -> shared memory [to be used for reduction]
    //
    //  error_ += (source.points_[corres[threadIdx][0]] -
    //              target.points_[corres[threadIdx][1]]).squaredNorm();
    //
    // Existing CPU Implementation:
    // for (const auto &c : corres) {
    //     err += (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
    // }

    // error = (source.IndexGet({corres[0]}) -
    // target.IndexGet({corres[1]}))*(source.IndexGet({corres[0]}) -
    // target.IndexGet({corres[1]}))
    //
    // return std::sqrt(err / (double)corres.size());
    return err;
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // if (corres.empty()) return core::Tensor::Eye(4,
    //                              core::Dtype::Float64,
    //                              core::Device("CPU:0"));

    // Existing: Creates a new Cx3 src and target points based on correspondaces
    // New: Use Advanced indexing
    //
    // core::Tensor source_mat(3, corres.size());
    // core::Tensor target_mat(3, corres.size());
    // for (size_t i = 0; i < corres.size(); i++) {
    //     source_mat.block<3, 1>(0, i) = source.points_[corres[i][0]];
    //     target_mat.block<3, 1>(0, i) = target.points_[corres[i][1]];
    // }

    // Tensor::umeyama to be implemented for calculating Transformation
    //
    // return Eigen::umeyama(source_mat, target_mat, with_scaling_);

    // Temp. for finalizing header file. Return Type: Matrix4d
    return core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0"));
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // if (corres.empty() || !target.HasPointNormals()) return 0.0;
    double err = 0.0;
    // double r;

    // Vectorisation:
    //  source, target -> shared memory
    //  error_ -> shared memory [to be used for reduction]
    //
    //  error_ += (source.points_[corres[threadIdx][0]] -
    //               target.points_[corres[threadIdx][1]]).dot(
    //                      target.points_[corres[threadIdx][1]])
    //
    // Existing CPU Implementation:
    // for (const auto &c : corres) {
    //     r = (source.points_[c[0]] - target.points_[c[1]])
    //                 .dot(target.normals_[c[1]]);
    //     err += r * r;
    // }
    // return std::sqrt(err / (double)corres.size());
    return err;
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // if (corres.empty() || !target.HasPointNormals())
    //     return core::Tensor::Eye(4,
    //                          core::Dtype::Float64, core::Device("CPU:0"));

    // auto compute_jacobian_and_residual = [&](int i, Eigen::Vector6d &J_r,
    //                                          double &r, double &w) {
    //     const core::Tensor &vs = source.points_[corres[i][0]];
    //     const core::Tensor &vt = target.points_[corres[i][1]];
    //     const core::Tensor &nt = target.normals_[corres[i][1]];
    //     r = (vs - vt).dot(nt);
    //     w = kernel_->Weight(r);
    //     J_r.block<3, 1>(0, 0) = vs.cross(nt);
    //     J_r.block<3, 1>(3, 0) = nt;
    // };

    // core::Tensor JTJ;
    // core::Tensor JTr;
    // double r2;
    // std::tie(JTJ, JTr, r2) =
    //         utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
    //                 compute_jacobian_and_residual, (int)corres.size());

    // bool is_success;
    // core::Tensor extrinsic;
    // std::tie(is_success, extrinsic) =
    //         utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    // return is_success ? extrinsic : core::Tensor::Eye(4,
    //                              core::Dtype::Float64,
    //                              core::Device("CPU:0"));
    return core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0"));
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
