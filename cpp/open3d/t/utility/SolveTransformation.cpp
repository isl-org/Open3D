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

#include "open3d/t/utility/SolveTransformation.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace t {
namespace utility {

core::Tensor ComputeTransformationFromRt(const core::Tensor &R,
                                         const core::Tensor &t,
                                         const core::Dtype &dtype,
                                         const core::Device &device) {
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    R.AssertShape({3, 3});
    R.AssertDevice(device);
    t.AssertShape({3});
    t.AssertDevice(device);

    // Rotation
    core::Tensor translate = t.Copy().Reshape({1, 3});
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(0, 3, 1)},
            R);
    // Translation and Scale [Assumed to be 1]
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            t.Reshape({3, 1}));
    transformation[3][3] = 1;
    return transformation;
}

double det_(const core::Tensor D) {
    // TODO: Create a proper op for Determinant
    D.AssertShape({3, 3});
    core::Tensor D_ = D.Copy();
    D_[0][0] = D_[0][0] * (D_[1][1] * D_[2][2] - D_[1][2] * D_[2][1]);
    D_[0][1] = D_[0][1] * (D_[1][0] * D_[2][2] - D_[2][0] * D_[1][2]);
    D_[0][2] = D_[0][2] * (D_[1][0] * D_[2][1] - D_[2][0] * D_[1][1]);
    D_[0][0] = D_[0][0] - D_[0][1] + D_[0][2];
    return D_[0][0].Item<float>();
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
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            X.GetItem({core::TensorKey::Slice(3, 6, 1)}).Reshape({3, 1}));

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
        open3d::utility::LogError(
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

}  // namespace utility
}  // namespace t
}  // namespace open3d
