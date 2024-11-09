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

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/pointnet/SamplingKernel.h"
#include "paddle/extension.h"

#ifdef BUILD_CUDA_MODULE

std::vector<paddle::Tensor> FurthestPointSampling(paddle::Tensor &points,
                                                  const int64_t sample_size) {
    int batch_size = points.shape()[0];
    int pts_size = points.shape()[1];

    auto place = points.place();
    paddle::Tensor out =
            paddle::full({batch_size, sample_size}, 0,
                         paddle::DataType(ToPaddleDtype<int>()), place);
    paddle::Tensor temp =
            paddle::full({batch_size, pts_size}, 1e10,
                         paddle::DataType(ToPaddleDtype<float>()), place);

    const float *points_data = points.data<float>();
    float *temp_data = temp.data<float>();
    int *out_data = out.data<int>();

    furthest_point_sampling_launcher(
            batch_size, pts_size, sample_size, points_data, temp_data, out_data,
            reinterpret_cast<uint64_t>(points.stream()));

    return {out};
}

std::vector<paddle::DataType> FurthestPointSamplingInferDtype() {
    return {paddle::DataType::INT32};
}

std::vector<std::vector<int64_t>> FurthestPointSamplingInferShape(
        std::vector<int64_t> points_shape, const int64_t sample_size) {
    return {{points_shape[0], sample_size}};
}

PD_BUILD_OP(open3d_furthest_point_sampling)
        .Inputs({"points"})
        .Outputs({"out"})
        .Attrs({
                "sample_size: int64_t",
        })
        .SetKernelFn(PD_KERNEL(FurthestPointSampling))
        .SetInferShapeFn(PD_INFER_SHAPE(FurthestPointSamplingInferShape))
        .SetInferDtypeFn(PD_INFER_DTYPE(FurthestPointSamplingInferDtype));

#endif
