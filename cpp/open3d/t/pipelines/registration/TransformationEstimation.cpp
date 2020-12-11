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

double det_(const core::Tensor &D) {
    // TODO: Create a proper op for Determinant
    D.AssertShape({3, 3});
    core::Tensor D_ = D.Copy();
    D_[0][0] = D_[0][0] * (D_[1][1] * D_[2][2] - D_[1][2] * D_[2][1]);
    D_[0][1] = D_[0][1] * (D_[1][0] * D_[2][2] - D_[2][0] * D_[1][2]);
    D_[0][2] = D_[0][2] * (D_[1][0] * D_[2][1] - D_[2][0] * D_[1][1]);
    D_[0][0] = D_[0][0] - D_[0][1] + D_[0][2];
    return D_[0][0].Item<float>();
}

core::Tensor ComputeTransformationFromRt(const core::Tensor &R,
                                         const core::Tensor &t,
                                         const core::Dtype &dtype,
                                         const core::Device &device) {
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    R.AssertShape({3, 3});
    R.AssertDevice(device);
    t.AssertShape({3, 1});
    t.AssertDevice(device);

    // Rotation
    core::Tensor translate = t.Copy().Reshape({1, 3});
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(0, 3, 1)},
            R);
    // Translation and Scale [Assumed to be 1]
    transformation[0][3] = t[0];
    transformation[1][3] = t[1];
    transformation[2][3] = t[2];
    transformation[3][3] = 1;
    return transformation;
}

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    double error;

    core::Tensor select_bool = (corres.Ne(-1)).Reshape({-1});
    core::Tensor source_select = source.GetPoints().IndexGet({select_bool});
    core::Tensor corres_select = corres.IndexGet({select_bool}).Reshape({-1});
    core::Tensor target_select = target.GetPoints().IndexGet({corres_select});

    core::Tensor error_t = (source_select - target_select);
    error_t.Mul_(error_t);
    error = error_t.Sum({0, 1}).Item<double_t>();
    return std::sqrt(error / (double)corres_select.GetShape()[0]);
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    // Assert PointCloud to have Float32 Dtype,
    // as the same is required by SVD Solver.

    // Remove hardcoded dtype
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor select_bool = (corres.Ne(-1)).Reshape({-1});
    core::Tensor source_select =
            source.GetPoints().IndexGet({select_bool}).To(dtype);
    core::Tensor corres_select = corres.IndexGet({select_bool}).Reshape({-1});
    core::Tensor target_select =
            target.GetPoints().IndexGet({corres_select}).To(dtype);

    // https://ieeexplore.ieee.org/document/88573

    core::Tensor mux = source_select.Mean({0}, true).To(dtype);
    core::Tensor muy = target_select.Mean({0}, true).To(dtype);

    core::Tensor Sxy = ((target_select - muy)
                                .T()
                                .Matmul(source_select - mux)
                                .Div_((double)corres_select.GetShape()[0])
                                .To(dtype));

    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (det_(U) * det_(VT.T()) < 0) S[-1][-1] = -1;

    core::Tensor R, t;
    R = U.Matmul(S.Matmul(VT)).To(dtype);
    t = muy.Reshape({-1}) - R.Matmul(mux.T()).Reshape({-1}).To(dtype);

    return ComputeTransformationFromRt(R, t, dtype, device)
            .To(core::Dtype::Float64);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // TODO: Source Target Device Assert has been so frequently used,
    // that an op can be defined for this.
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    if (!target.HasPointNormals()) return 0.0;

    core::Tensor select_bool = (corres.Ne(-1)).Reshape({-1});
    core::Tensor source_select = source.GetPoints().IndexGet({select_bool});
    core::Tensor corres_select = corres.IndexGet({select_bool}).Reshape({-1});
    core::Tensor target_select = target.GetPoints().IndexGet({corres_select});
    core::Tensor target_n_select =
            target.GetPointNormals().IndexGet({corres_select});

    core::Tensor error_t =
            (source_select - target_select).Matmul(target_n_select);
    error_t.Mul_(error_t);
    double error = error_t.Sum({0, 1}).Item<double_t>();
    return std::sqrt(error / (double)corres_select.GetShape()[0]);
}

core::Tensor ComputeTransformationFromPose(const core::Tensor &X,
                                           const core::Dtype &dtype,
                                           const core::Device &device) {
    // TODO:
    // A better implementation of this function
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    X.AssertShape({6});
    X.AssertDevice(device);

    // Rotation from Pose X
    transformation[0][0] = X[2].Cos().Mul(X[1].Cos());
    transformation[0][1] =
            -1 * X[2].Sin() * X[0].Cos() + X[2].Cos() * X[1].Sin() * X[0].Sin();
    transformation[0][2] =
            X[2].Sin() * X[0].Sin() + X[2].Cos() * X[1].Sin() * X[0].Cos();
    transformation[1][0] = X[2].Sin() * X[1].Cos();
    transformation[1][1] =
            X[2].Cos() * X[0].Cos() + X[2].Sin() * X[1].Sin() * X[0].Sin();
    transformation[1][2] =
            -1 * X[2].Cos() * X[0].Sin() + X[2].Sin() * X[1].Sin() * X[0].Cos();
    transformation[2][0] = -1 * X[1].Sin();
    transformation[2][1] = X[1].Cos() * X[0].Sin();
    transformation[2][2] = X[1].Cos() * X[0].Cos();

    // Translation from Pose X
    transformation[0][3] = X[3];
    transformation[1][3] = X[4];
    transformation[2][3] = X[5];

    // Current Implementation DOES NOT SUPPORT SCALE transfomation
    transformation[3][3] = 1;
    return transformation;
}

core::Tensor Compute_A(const core::Tensor &source_select,
                       const core::Tensor &target_n_select,
                       const core::Dtype &dtype,
                       const core::Device &device) {
    // TODO:
    // A better implementation of this function
    source_select.AssertDevice(device);
    target_n_select.AssertDevice(device);
    if (target_n_select.GetShape() != source_select.GetShape()) {
        utility::LogError(
                " [Compute_A:] Target Normal Pointcloud Correspondace Shape "
                " {} != Corresponding Source Pointcloud's Shape {}.",
                target_n_select.GetShape().ToString(),
                source_select.GetShape().ToString());
    }

    auto num_corres = source_select.GetShape()[0];
    // if num_corres == 0 : LogError / Return 0 Tensor

    // Slicing Normals: (nx, ny, nz) and Source Points: (sx, sy, sz)
    core::Tensor nx =
            target_n_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                     core::TensorKey::Slice(0, 1, 1)});
    core::Tensor ny =
            target_n_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                     core::TensorKey::Slice(1, 2, 1)});
    core::Tensor nz =
            target_n_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                     core::TensorKey::Slice(2, 3, 1)});
    core::Tensor sx =
            source_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                   core::TensorKey::Slice(0, 1, 1)});
    core::Tensor sy =
            source_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                   core::TensorKey::Slice(1, 2, 1)});
    core::Tensor sz =
            source_select.GetItem({core::TensorKey::Slice(0, num_corres, 1),
                                   core::TensorKey::Slice(2, 3, 1)});

    // Cross Product Calculation
    core::Tensor a1 = (nz * sy) - (ny * sz);
    core::Tensor a2 = (nx * sz) - (nz * sx);
    core::Tensor a3 = (ny * sx) - (nx * sy);

    // Putting the pieces back together.
    core::Tensor A({num_corres, 6}, dtype, device);
    A.SetItem({core::TensorKey::Slice(0, num_corres, 1),
               core::TensorKey::Slice(0, 1, 1)},
              a1);
    A.SetItem({core::TensorKey::Slice(0, num_corres, 1),
               core::TensorKey::Slice(1, 2, 1)},
              a2);
    A.SetItem({core::TensorKey::Slice(0, num_corres, 1),
               core::TensorKey::Slice(2, 3, 1)},
              a3);
    A.SetItem({core::TensorKey::Slice(0, num_corres, 1),
               core::TensorKey::Slice(3, 6, 1)},
              target_n_select);
    return A;
}

core::Tensor SolvePointToPlaneTransformation(const geometry::PointCloud &source,
                                             const geometry::PointCloud &target,
                                             const core::Tensor &corres,
                                             const core::Dtype dtype) {
    // TODO:
    // Asserts and Checks
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    // A better implementation of this function
    core::Tensor select_bool = (corres.Ne(-1)).Reshape({-1});
    core::Tensor source_select = source.GetPoints().IndexGet({select_bool});
    core::Tensor corres_select = corres.IndexGet({select_bool}).Reshape({-1});
    core::Tensor target_select = target.GetPoints().IndexGet({corres_select});
    core::Tensor target_n_select =
            target.GetPointNormals().IndexGet({corres_select});

    // TODO: SANITY CHECKS
    core::Tensor B = ((target_select - source_select).Mul_(target_n_select))
                             .Sum({1}, true);
    core::Tensor A = Compute_A(source_select, target_n_select, dtype, device);
    core::Tensor X = (A.LeastSquares(B)).Reshape({-1});
    return ComputeTransformationFromPose(X, dtype, device);
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &corres) const {
    // TODO: if corres empty throw Error
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    // Remove hardcoded dtype
    core::Dtype dtype = core::Dtype::Float64;
    return SolvePointToPlaneTransformation(source, target, corres, dtype);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
