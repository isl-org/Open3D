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
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    double error;
    // TODO: Revist to support Float32 and 64 without type conversion.
    core::Tensor source_select = source.GetPoints().IndexGet({corres.first});
    core::Tensor target_select = target.GetPoints().IndexGet({corres.second});

    core::Tensor error_t = (source_select - target_select);
    error_t.Mul_(error_t);
    error = static_cast<double>(error_t.Sum({0, 1}).Item<float>());
    return std::sqrt(error / static_cast<double>(corres.second.GetShape()[0]));
}

core::Tensor TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }
    core::Tensor source_select = source.GetPoints().IndexGet({corres.first});
    core::Tensor target_select = target.GetPoints().IndexGet({corres.second});

    // https://ieeexplore.ieee.org/document/88573
    core::Tensor mux = source_select.Mean({0}, true);
    core::Tensor muy = target_select.Mean({0}, true);
    core::Tensor Sxy =
            ((target_select - muy)
                     .T()
                     .Matmul(source_select - mux)
                     .Div_(static_cast<float>(corres.second.GetShape()[0])));
    core::Tensor U, D, VT;
    std::tie(U, D, VT) = Sxy.SVD();
    core::Tensor S = core::Tensor::Eye(3, dtype, device);
    if (U.Det() * (VT.T()).Det() < 0) {
        S[-1][-1] = -1;
    }
    core::Tensor R, t;
    R = U.Matmul(S.Matmul(VT));
    t = muy.Reshape({-1}) - R.Matmul(mux.T()).Reshape({-1});

    return t::pipelines::kernel::RtToTransformation(R, t);
}

double TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
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
            (source_select - target_select).Mul_(target_n_select);
    error_t.Mul_(error_t);
    double error = static_cast<double>(error_t.Sum({0, 1}).Item<float>());
    return std::sqrt(error / static_cast<double>(corres.second.GetShape()[0]));
}

core::Tensor TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        CorrespondenceSet &corres) const {
    // TODO: If corres empty throw Error.
    core::Device device = source.GetDevice();
    core::Dtype dtype = core::Dtype::Float32;
    source.GetPoints().AssertDtype(dtype);
    target.GetPoints().AssertDtype(dtype);
    if (target.GetDevice() != device) {
        utility::LogError(
                "Target Pointcloud device {} != Source Pointcloud's device {}.",
                target.GetDevice().ToString(), device.ToString());
    }

    core::Tensor source_select =
            source.GetPoints().IndexGet({corres.first}).To(dtype);
    core::Tensor target_select =
            target.GetPoints().IndexGet({corres.second}).To(dtype);
    core::Tensor target_n_select =
            target.GetPointNormals().IndexGet({corres.second}).To(dtype);

    core::Tensor B = ((target_select - source_select).Mul_(target_n_select))
                             .Sum({1}, true)
                             .To(dtype);

    // --- Computing A in AX = B.
    int64_t num_corres = source_select.GetShape()[0];
    // Slicing normals: (nx, ny, nz) and source points: (sx, sy, sz).
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
    // Cross product calculation.
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

    core::Tensor Pose = (A.LeastSquares(B)).Reshape({-1}).To(dtype);
    return t::pipelines::kernel::PoseToTransformation(Pose);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
