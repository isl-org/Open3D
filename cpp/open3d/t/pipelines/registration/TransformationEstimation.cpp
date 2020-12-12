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

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

double TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    double error;
    core::Tensor source_select = source.GetPoints().IndexGet({corres.first});
    core::Tensor target_select = target.GetPoints().IndexGet({corres.second});

    core::Tensor error_t = (source_select - target_select);
    error_t.Mul_(error_t);
    error = error_t.Sum({0, 1}).Item<double_t>();
    return std::sqrt(error / (double)corres.second.GetShape()[0]);
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    // Assert PointCloud to have Float32 Dtype,
    // as the same is required by SVD Solver

    // Remove hardcoded dtype
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor source_select =
            source.GetPoints().IndexGet({corres.first}).To(dtype);
    core::Tensor target_select =
            target.GetPoints().IndexGet({corres.second}).To(dtype);

    // https://ieeexplore.ieee.org/document/88573

    core::Tensor mux = source_select.Mean({0}, true).To(dtype);
    core::Tensor muy = target_select.Mean({0}, true).To(dtype);
    core::Tensor Sxy = ((target_select - muy)
                                .T()
                                .Matmul(source_select - mux)
                                .Div_((float)corres.second.GetShape()[0])
                                .To(dtype));
    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (det_(U) * det_(VT.T()) < 0) {
        S[-1][-1] = -1;
    }
    core::Tensor R, t;
    R = U.Matmul(S.Matmul(VT)).To(dtype);
    t = muy.Reshape({-1}) - R.Matmul(mux.T()).Reshape({-1}).To(dtype);

    return ComputeTransformationFromRt(R, t, dtype, device).To(dtype);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    // TODO: Source Target Device Assert has been so frequently used,
    // that an op can be defined for this.
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    if (!target.HasPointNormals()) return 0.0;

    core::Tensor source_select = source.GetPoints().IndexGet({corres.first});
    core::Tensor target_select = target.GetPoints().IndexGet({corres.second});
    core::Tensor target_n_select =
            target.GetPointNormals().IndexGet({corres.second});

    core::Tensor error_t =
            (source_select - target_select).Matmul(target_n_select);
    error_t.Mul_(error_t);
    double error = error_t.Sum({0, 1}).Item<double_t>();
    return std::sqrt(error / (double)corres.second.GetShape()[0]);
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    // TODO: if corres empty throw Error
    core::Device device = source.GetDevice();
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    // Remove hardcoded dtype
    core::Dtype dtype = core::Dtype::Float32;
    return SolvePointToPlaneTransformation(source, target, corres, dtype);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
