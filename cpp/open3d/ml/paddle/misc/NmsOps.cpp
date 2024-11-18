// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/ml/contrib/Nms.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> Nms(paddle::Tensor& boxes,
                                paddle::Tensor& scores,
                                double nms_overlap_thresh) {
    CHECK_TYPE(boxes, phi::DataType::FLOAT32);
    CHECK_TYPE(scores, phi::DataType::FLOAT32);

    std::vector<int64_t> keep_indices_blob;
    if (boxes.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        keep_indices_blob = open3d::ml::contrib::NmsCUDAKernel(
                boxes.data<float>(), scores.data<float>(), boxes.shape()[0],
                nms_overlap_thresh);
#else
        PD_CHECK(false, "Nms was not compiled with CUDA support");

#endif
    } else {
        keep_indices_blob = open3d::ml::contrib::NmsCPUKernel(
                boxes.data<float>(), scores.data<float>(), boxes.shape()[0],
                nms_overlap_thresh);
    }

    paddle::IntArray out_shape(
            {static_cast<int64_t>(keep_indices_blob.size())});
    paddle::IntArray out_strides({1});
    // NOTE: Not pass deleter because data will be free as vector destroy.
    if (keep_indices_blob.data()) {
        paddle::Tensor temp_keep_indices = paddle::from_blob(
                keep_indices_blob.data(), out_shape, out_strides,
                phi::DataType::INT64, phi::DataLayout::NCHW, phi::CPUPlace());
        if (boxes.is_gpu()) {
            temp_keep_indices = temp_keep_indices.copy_to(boxes.place(), false);
        }

        paddle::Tensor keep_indices = paddle::empty_like(temp_keep_indices);

        return {paddle::experimental::copysign(temp_keep_indices,
                                               keep_indices)};
    } else {
        // keep indices is nullptr
        return {InitializedEmptyTensor<int64_t>({0}, boxes.place())};
    }
}

std::vector<paddle::DataType> NmsInferDtype() {
    return {paddle::DataType::INT64};
}

PD_BUILD_OP(open3d_nms)
        .Inputs({"boxes", "scores"})
        .Outputs({"keep_indices"})
        .Attrs({"nms_overlap_thresh: double"})
        .SetKernelFn(PD_KERNEL(Nms))
        .SetInferDtypeFn(PD_INFER_DTYPE(NmsInferDtype));
