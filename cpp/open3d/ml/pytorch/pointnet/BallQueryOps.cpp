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

#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/pointnet/BallQueryKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
torch::Tensor ball_query(torch::Tensor xyz,
                         torch::Tensor center,
                         double radius,
                         const int64_t nsample) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int ball_num = center.size(1);

    auto device = xyz.device();
    torch::Tensor out =
            torch::zeros({batch_size, ball_num, nsample},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    const float *center_data = center.data_ptr<float>();
    const float *xyz_data = xyz.data_ptr<float>();
    int *idx = out.data_ptr<int>();

    ball_query_launcher(batch_size, pts_num, ball_num, radius, nsample,
                        center_data, xyz_data, idx);
    return out;
}

static auto registry = torch::RegisterOperators(
        "open3d::ball_query(Tensor xyz, Tensor center,"
        "float radius, int nsample)"
        " -> Tensor out",
        &ball_query);
#endif
