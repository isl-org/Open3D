// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
#include "open3d/ml/paddle/PaddleHelper.h"

#ifdef BUILD_CUDA_MODULE

std::vector<paddle::Tensor> RoiPool(paddle::Tensor &xyz,
                                    paddle::Tensor &boxes3d,
                                    paddle::Tensor &pts_feature,
                                    const int64_t sampled_pts_num) {
    int batch_size = xyz.shape()[0];
    int pts_num = xyz.shape()[1];
    int boxes_num = boxes3d.shape()[1];
    int feature_in_len = pts_feature.shape()[2];

    auto place = xyz.place();
    paddle::Tensor features = paddle::full(
            {batch_size, boxes_num, sampled_pts_num, 3 + feature_in_len}, 0.0f,
            paddle::DataType(ToPaddleDtype<float>()), place);

    paddle::Tensor empty_flag =
            paddle::full({batch_size, boxes_num}, 0.0f,
                         paddle::DataType(ToPaddleDtype<int>()), place);

    const float *xyz_data = xyz.data<float>();
    const float *boxes3d_data = boxes3d.data<float>();
    const float *pts_feature_data = pts_feature.data<float>();
    float *pooled_features_data = features.data<float>();
    int *pooled_empty_flag_data = empty_flag.data<int>();

    open3d::ml::contrib::roipool3dLauncher(
            batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
            xyz_data, boxes3d_data, pts_feature_data, pooled_features_data,
            pooled_empty_flag_data);

    return {features, empty_flag};
}

std::vector<paddle::DataType> RoiPoolInferDtype() {
    return {paddle::DataType::FLOAT32, paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> RoiPoolInferShape(
        std::vector<int64_t> xyz_shape,
        std::vector<int64_t> boxes3d_shape,
        std::vector<int64_t> pts_feature_shape,
        const int64_t sampled_pts_num) {
    std::vector<int64_t> features_shape{xyz_shape[0], boxes3d_shape[1],
                                        sampled_pts_num,
                                        3 + pts_feature_shape[2]};
    return {features_shape, {xyz_shape[0], boxes3d_shape[1]}};
}

PD_BUILD_OP(open3d_roi_pool)
        .Inputs({"xyz", "boxes3d", "pts_feature"})
        .Outputs({"features", "empty_flag"})
        .Attrs({
                "sampled_pts_num: int64_t",
        })
        .SetKernelFn(PD_KERNEL(RoiPool))
        .SetInferShapeFn(PD_INFER_SHAPE(RoiPoolInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(RoiPoolInferDtype));

#endif
