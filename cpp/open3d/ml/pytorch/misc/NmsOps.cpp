// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/ml/contrib/Nms.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

torch::Tensor Nms(torch::Tensor boxes,
                  torch::Tensor scores,
                  double nms_overlap_thresh) {
    boxes = boxes.contiguous();
    CHECK_TYPE(boxes, kFloat);
    CHECK_TYPE(scores, kFloat);

    if (boxes.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        std::vector<int64_t> keep_indices = open3d::ml::contrib::NmsCUDAKernel(
                boxes.data_ptr<float>(), scores.data_ptr<float>(),
                boxes.size(0), nms_overlap_thresh);
        return torch::from_blob(keep_indices.data(),
                                {static_cast<int64_t>(keep_indices.size())},
                                torch::TensorOptions().dtype(torch::kLong))
                .to(boxes.device());
#else
        TORCH_CHECK(false, "Nms was not compiled with CUDA support")

#endif
    } else {
        std::vector<int64_t> keep_indices = open3d::ml::contrib::NmsCPUKernel(
                boxes.data_ptr<float>(), scores.data_ptr<float>(),
                boxes.size(0), nms_overlap_thresh);
        return torch::from_blob(keep_indices.data(),
                                {static_cast<int64_t>(keep_indices.size())},
                                torch::TensorOptions().dtype(torch::kLong))
                .clone();
    }
}

static auto registry = torch::RegisterOperators(
        "open3d::nms(Tensor boxes, Tensor scores, float "
        "nms_overlap_thresh) -> "
        "Tensor keep_indices",
        &Nms);
