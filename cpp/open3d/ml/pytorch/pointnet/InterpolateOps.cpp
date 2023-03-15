// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on Pointnet2 Library (MIT License):
//    https://github.com/sshaoshuai/Pointnet2.PyTorch
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

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/pointnet/InterpolateKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
std::tuple<torch::Tensor, torch::Tensor> three_nn(torch::Tensor query_pts,
                                                  torch::Tensor data_pts) {
    int batch_size = query_pts.size(0);
    int pts_num_out = query_pts.size(1);
    int pts_num_in = data_pts.size(1);

    auto device = data_pts.device();
    torch::Tensor out_idx =
            torch::zeros({batch_size, pts_num_out, 3},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    torch::Tensor out_dist2 =
            torch::zeros({batch_size, pts_num_out, 3},
                         torch::dtype(ToTorchDtype<float>()).device(device));

    const float *pts_out = query_pts.data_ptr<float>();
    const float *pts_in = data_pts.data_ptr<float>();
    float *dist2 = out_dist2.data_ptr<float>();
    int *idx = out_idx.data_ptr<int>();

    three_nn_launcher(batch_size, pts_num_out, pts_num_in, pts_out, pts_in,
                      dist2, idx);

    return std::tuple<torch::Tensor, torch::Tensor>(out_dist2, out_idx);
}

torch::Tensor three_interpolate(torch::Tensor points,
                                torch::Tensor idx,
                                torch::Tensor weights) {
    int batch_size = points.size(0);
    int C = points.size(1);
    int M = points.size(2);
    int N = idx.size(1);

    auto device = points.device();
    torch::Tensor out =
            torch::zeros({batch_size, C, N},
                         torch::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data_ptr<float>();
    const float *weights_data = weights.data_ptr<float>();
    const int *idx_data = idx.data_ptr<int>();
    float *out_data = out.data_ptr<float>();

    three_interpolate_launcher(batch_size, C, M, N, points_data, idx_data,
                               weights_data, out_data);

    return out;
}

torch::Tensor three_interpolate_grad(torch::Tensor grad_out,
                                     torch::Tensor idx,
                                     torch::Tensor weights,
                                     const int64_t M) {
    int batch_size = grad_out.size(0);
    int C = grad_out.size(1);
    int N = grad_out.size(2);

    auto device = grad_out.device();
    torch::Tensor out =
            torch::zeros({batch_size, C, M},
                         torch::dtype(ToTorchDtype<float>()).device(device));

    const float *grad_out_data = grad_out.data_ptr<float>();
    const float *weights_data = weights.data_ptr<float>();
    const int *idx_data = idx.data_ptr<int>();

    float *out_data = out.data_ptr<float>();

    three_interpolate_grad_launcher(batch_size, C, N, M, grad_out_data,
                                    idx_data, weights_data, out_data);

    return out;
}

static auto registry_nn = torch::RegisterOperators(
        "open3d::three_nn(Tensor query_pts, Tensor data_pts)"
        " -> (Tensor dist, Tensor idx)",
        &three_nn);

static auto registry = torch::RegisterOperators(
        "open3d::three_interpolate(Tensor points,"
        "Tensor idx, Tensor weights)"
        " -> Tensor out",
        &three_interpolate);

static auto registry_grad = torch::RegisterOperators(
        "open3d::three_interpolate_grad(Tensor points,"
        "Tensor idx, Tensor weights, int N)"
        " -> Tensor out",
        &three_interpolate_grad);
#endif
