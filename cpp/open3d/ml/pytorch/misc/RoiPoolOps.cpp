// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on PointRCNN Library (MIT License):
//    https://github.com/sshaoshuai/PointRCNN
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

#include "open3d/ml/contrib/RoiPoolKernel.h"
#include "open3d/ml/pytorch/TorchHelper.h"

#ifdef BUILD_CUDA_MODULE
std::tuple<torch::Tensor, torch::Tensor> roi_pool(
        torch::Tensor xyz,
        torch::Tensor boxes3d,
        torch::Tensor pts_feature,
        const int64_t sampled_pts_num) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int boxes_num = boxes3d.size(1);
    int feature_in_len = pts_feature.size(2);

    auto device = xyz.device();
    torch::Tensor features = torch::zeros(
            {batch_size, boxes_num, sampled_pts_num, 3 + feature_in_len},
            torch::dtype(ToTorchDtype<float>()).device(device));

    torch::Tensor empty_flag =
            torch::zeros({batch_size, boxes_num},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    const float *xyz_data = xyz.data_ptr<float>();
    const float *boxes3d_data = boxes3d.data_ptr<float>();
    const float *pts_feature_data = pts_feature.data_ptr<float>();
    float *pooled_features_data = features.data_ptr<float>();
    int *pooled_empty_flag_data = empty_flag.data_ptr<int>();

    open3d::ml::contrib::roipool3dLauncher(
            batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
            xyz_data, boxes3d_data, pts_feature_data, pooled_features_data,
            pooled_empty_flag_data);

    return std::tuple<torch::Tensor, torch::Tensor>(features, empty_flag);
}

static auto registry = torch::RegisterOperators(
        "open3d::roi_pool(Tensor xyz, Tensor boxes3d,"
        "Tensor pts_feature, int sampled_pts_num)"
        " -> (Tensor features, Tensor flags)",
        &roi_pool);
#endif
