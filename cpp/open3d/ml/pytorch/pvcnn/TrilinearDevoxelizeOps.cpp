// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
//
//    Based on PVCNN Library (MIT License):
//    https://github.com/mit-han-lab/pvcnn
//
// Copyright (c) 2018 Zhijian Liu, Haotian Tang, Yujun Lin
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/pvcnn/TrilinearDevoxelizeKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
std::vector<at::Tensor> trilinear_devoxelize_forward(
        const int64_t r,
        const bool is_training,
        const at::Tensor coords,
        const at::Tensor features) {
    CHECK_CUDA(features);
    CHECK_CUDA(coords);
    CHECK_CONTIGUOUS(features);
    CHECK_CONTIGUOUS(coords);
    CHECK_TYPE(features, kFloat32);
    CHECK_TYPE(coords, kFloat32);

    CHECK_SAME_DTYPE(features, coords);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim batch_size("batch_size");
        Dim feat_dim("feat_dim");
        Dim num_points("num_points");
        Dim resolution("resolution");
        CHECK_SHAPE(coords, batch_size, 3, num_points);
        CHECK_SHAPE(features, batch_size, feat_dim, resolution, resolution,
                    resolution);
    }

    int b = features.size(0);
    int c = features.size(1);
    int n = coords.size(2);
    int r2 = r * r;
    int r3 = r2 * r;
    at::Tensor outs = torch::zeros(
            {b, c, n},
            at::device(features.device()).dtype(at::ScalarType::Float));
    if (is_training) {
        at::Tensor inds = torch::zeros(
                {b, 8, n},
                at::device(features.device()).dtype(at::ScalarType::Int));
        at::Tensor wgts = torch::zeros(
                {b, 8, n},
                at::device(features.device()).dtype(at::ScalarType::Float));
        TrilinearDevoxelize(b, c, n, r, r2, r3, true, coords.data_ptr<float>(),
                            features.data_ptr<float>(), inds.data_ptr<int>(),
                            wgts.data_ptr<float>(), outs.data_ptr<float>());
        return {outs, inds, wgts};
    } else {
        at::Tensor inds = torch::zeros(
                {1}, at::device(features.device()).dtype(at::ScalarType::Int));
        at::Tensor wgts = torch::zeros(
                {1},
                at::device(features.device()).dtype(at::ScalarType::Float));
        TrilinearDevoxelize(b, c, n, r, r2, r3, false, coords.data_ptr<float>(),
                            features.data_ptr<float>(), inds.data_ptr<int>(),
                            wgts.data_ptr<float>(), outs.data_ptr<float>());
        return {outs, inds, wgts};
    }
}

at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights,
                                         const int64_t r) {
    CHECK_CUDA(grad_y);
    CHECK_CUDA(weights);
    CHECK_CUDA(indices);
    CHECK_CONTIGUOUS(grad_y);
    CHECK_CONTIGUOUS(weights);
    CHECK_CONTIGUOUS(indices);
    CHECK_TYPE(indices, kInt32);
    CHECK_TYPE(grad_y, kFloat32);
    CHECK_TYPE(weights, kFloat32);

    CHECK_SAME_DTYPE(weights, grad_y);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim batch_size("batch_size");
        Dim feat_dim("feat_dim");
        Dim num_points("num_points");
        CHECK_SHAPE(grad_y, batch_size, feat_dim, num_points);
        CHECK_SHAPE(indices, batch_size, 8, num_points);
        CHECK_SHAPE(weights, batch_size, 8, num_points);
    }

    int b = grad_y.size(0);
    int c = grad_y.size(1);
    int n = grad_y.size(2);
    int r3 = r * r * r;
    at::Tensor grad_x = torch::zeros(
            {b, c, r3},
            at::device(grad_y.device()).dtype(at::ScalarType::Float));
    TrilinearDevoxelizeGrad(b, c, n, r3, indices.data_ptr<int>(),
                            weights.data_ptr<float>(), grad_y.data_ptr<float>(),
                            grad_x.data_ptr<float>());
    return grad_x;
}

static auto registry = torch::RegisterOperators(
        "open3d::trilinear_devoxelize_forward(int r, bool is_training,"
        "Tensor coords, Tensor features)"
        " -> (Tensor[])",
        &trilinear_devoxelize_forward);

static auto registry_grad = torch::RegisterOperators(
        "open3d::trilinear_devoxelize_backward(Tensor grad_y,"
        "Tensor indices, Tensor weights, int r)"
        " -> Tensor grad_x",
        &trilinear_devoxelize_backward);

#endif
