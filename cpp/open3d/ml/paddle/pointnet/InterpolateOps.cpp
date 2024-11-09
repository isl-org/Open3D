// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on Pointnet2 Library (MIT License):
//    https://github.com/sshaoshuai/Pointnet2.PyPaddle
//
//    Copyright (c) 2019 Shaoshuai Shi
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files (the "Software"),
//    to deal in the Software without restriction, including without limitation
//    the rights to use, copy, modify, merge, publish, distribute, sublicense,
//    and/or sell copies of the Software, and to permit persons to whom the
//    Software is furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
//***************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/pointnet/InterpolateKernel.h"
#include "paddle/extension.h"

#ifdef BUILD_CUDA_MODULE
std::vector<paddle::Tensor> ThreeNN(paddle::Tensor &query_pts,
                                    paddle::Tensor &data_pts) {
    int batch_size = query_pts.shape()[0];
    int pts_num_out = query_pts.shape()[1];
    int pts_num_in = data_pts.shape()[1];

    auto place = data_pts.place();
    paddle::Tensor out_idx =
            paddle::full({batch_size, pts_num_out, 3}, 0,
                         paddle::DataType(ToPaddleDtype<int>()), place);

    paddle::Tensor out_dist2 =
            paddle::zeros({batch_size, pts_num_out, 3},
                          paddle::DataType(ToPaddleDtype<float>()), place);

    const float *pts_out = query_pts.data<float>();
    const float *pts_in = data_pts.data<float>();
    float *dist2 = out_dist2.data<float>();
    int *idx = out_idx.data<int>();

    three_nn_launcher(batch_size, pts_num_out, pts_num_in, pts_out, pts_in,
                      dist2, idx,
                      reinterpret_cast<uint64_t>(query_pts.stream()));

    return {out_dist2, out_idx};
}

std::vector<paddle::DataType> ThreeNNInferDtype() {
    return {paddle::DataType::INT32, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> ThreeNNInferShape(
        std::vector<int64_t> query_pts_shape,
        std::vector<int64_t> data_pts_shape) {
    std::vector<int64_t> shape{query_pts_shape[0], query_pts_shape[1], 3};
    return {shape, shape};
}

PD_BUILD_OP(open3d_three_nn)
        .Inputs({"query_pts", "data_pts"})
        .Outputs({"dist", "idx"})
        .Attrs({})
        .SetKernelFn(PD_KERNEL(ThreeNN))
        .SetInferShapeFn(PD_INFER_SHAPE(ThreeNNInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(ThreeNNInferDtype));

std::vector<paddle::Tensor> ThreeInterpolate(paddle::Tensor &points,
                                             paddle::Tensor &idx,
                                             paddle::Tensor &weights) {
    int batch_size = points.shape()[0];
    int C = points.shape()[1];
    int M = points.shape()[2];
    int N = idx.shape()[1];

    auto place = points.place();
    paddle::Tensor out =
            paddle::full({batch_size, C, N}, 0.0f,
                         paddle::DataType(ToPaddleDtype<float>()), place);

    const float *points_data = points.data<float>();
    const float *weights_data = weights.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    three_interpolate_launcher(batch_size, C, M, N, points_data, idx_data,
                               weights_data, out_data,
                               reinterpret_cast<uint64_t>(points.stream()));

    return {out};
}

std::vector<paddle::DataType> ThreeInterpolateInferDtype() {
    return {paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> ThreeInterpolateInferShape(
        std::vector<int64_t> points_shape, std::vector<int64_t> idx_shape) {
    return {{points_shape[0], points_shape[1], idx_shape[1]}};
}

PD_BUILD_OP(open3d_three_interpolate)
        .Inputs({
                "points",
                "idx",
                "weights",
        })
        .Outputs({"out"})
        .Attrs({})
        .SetKernelFn(PD_KERNEL(ThreeInterpolate))
        .SetInferShapeFn(PD_INFER_SHAPE(ThreeInterpolateInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(ThreeInterpolateInferDtype));

std::vector<paddle::Tensor> ThreeInterpolateGrad(paddle::Tensor &grad_out,
                                                 paddle::Tensor &idx,
                                                 paddle::Tensor &weights,
                                                 const int64_t M) {
    int batch_size = grad_out.shape()[0];
    int C = grad_out.shape()[1];
    int N = grad_out.shape()[2];

    auto place = grad_out.place();
    paddle::Tensor out =
            paddle::full({batch_size, C, M}, 0.0f,
                         paddle::DataType(ToPaddleDtype<float>()), place);

    const float *grad_out_data = grad_out.data<float>();
    const float *weights_data = weights.data<float>();
    const int *idx_data = idx.data<int>();

    float *out_data = out.data<float>();

    three_interpolate_grad_launcher(
            batch_size, C, N, M, grad_out_data, idx_data, weights_data,
            out_data, reinterpret_cast<uint64_t>(grad_out.stream()));

    return {out};
}

std::vector<paddle::DataType> ThreeInterpolateGradInferDtype() {
    return {paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> ThreeInterpolateGradInferShape(
        std::vector<int64_t> grad_out_shape) {
    return {{grad_out_shape[0], grad_out_shape[1], grad_out_shape[2]}};
}

PD_BUILD_OP(open3d_three_interpolate_grad)
        .Inputs({"grad_out", "idx", "weights"})
        .Outputs({"out"})
        .Attrs({"M: int64_t"})
        .SetKernelFn(PD_KERNEL(ThreeInterpolateGrad))
        .SetInferShapeFn(PD_INFER_SHAPE(ThreeInterpolateGradInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(ThreeInterpolateGradInferDtype));

#endif
