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
#include "open3d/ml/pytorch/pointnet/SamplingKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE

torch::Tensor furthest_point_sampling(torch::Tensor points,
                                      const int64_t sample_size) {
    int batch_size = points.size(0);
    int pts_size = points.size(1);

    auto device = points.device();
    torch::Tensor out =
            torch::zeros({batch_size, sample_size},
                         torch::dtype(ToTorchDtype<int>()).device(device));
    torch::Tensor temp =
            torch::full({batch_size, pts_size}, 1e10,
                        torch::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data_ptr<float>();
    float *temp_data = temp.data_ptr<float>();
    int *out_data = out.data_ptr<int>();

    furthest_point_sampling_launcher(batch_size, pts_size, sample_size,
                                     points_data, temp_data, out_data);

    return out;
}

static auto registry_fp = torch::RegisterOperators(
        "open3d::furthest_point_sampling(Tensor points, int sample_siz)"
        " -> Tensor out",
        &furthest_point_sampling);
#endif
