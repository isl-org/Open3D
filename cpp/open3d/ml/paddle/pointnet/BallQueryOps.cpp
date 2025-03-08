// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/pointnet/BallQueryKernel.h"
#include "paddle/extension.h"

#ifdef BUILD_CUDA_MODULE

std::vector<paddle::Tensor> BallQuery(paddle::Tensor &xyz,
                                      paddle::Tensor &center,
                                      double radius,
                                      const int64_t nsample) {
    int batch_size = xyz.shape()[0];
    int pts_num = xyz.shape()[1];
    int ball_num = center.shape()[1];

    auto place = xyz.place();
    paddle::Tensor out =
            paddle::full({batch_size, ball_num, nsample}, 0.0f,
                         paddle::DataType(ToPaddleDtype<int>()), place);

    const float *center_data = center.data<float>();
    const float *xyz_data = xyz.data<float>();
    int *idx = out.data<int>();

    ball_query_launcher(batch_size, pts_num, ball_num, radius, nsample,
                        center_data, xyz_data, idx,
                        reinterpret_cast<uint64_t>(xyz.stream()));
    return {out};
}

std::vector<paddle::DataType> BallQueryInferDtype() {
    return {paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> BallQueryInferShape(
        std::vector<int64_t> xyz_shape,
        std::vector<int64_t> center_shape,
        const int64_t nsample) {
    return {{xyz_shape[0], xyz_shape[1], center_shape[1]}};
}

PD_BUILD_OP(open3d_ball_query)
        .Inputs({"xyz", "center"})
        .Outputs({"out"})
        .Attrs({
                "radius: double",
                "nsample: int64_t",
        })
        .SetKernelFn(PD_KERNEL(BallQuery))
        .SetInferShapeFn(PD_INFER_SHAPE(BallQueryInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(BallQueryInferDtype));

#endif
