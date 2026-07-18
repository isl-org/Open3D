// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/ml/contrib/Nms.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

#ifdef BUILD_SYCL_MODULE
#include <c10/xpu/XPUStream.h>
#endif

torch::Tensor Nms(torch::Tensor boxes,
                  torch::Tensor scores,
                  double nms_overlap_thresh) {
    boxes = boxes.contiguous();
    CHECK_TYPE(boxes, kFloat);
    CHECK_TYPE(scores, kFloat);

    if (boxes.is_cuda()) {
#ifdef BUILD_CUDA_MODULE
        // Write the kept indices directly into a tensor already on the CUDA
        // device: NmsCUDAKernel never touches the host for the (up to
        // n-sized) result data, only for the final small `count`, so there
        // is no device->host->device round trip.
        torch::Tensor keep_indices = torch::empty(
                {boxes.size(0)},
                torch::TensorOptions().dtype(torch::kLong).device(
                        boxes.device()));
        int count = open3d::ml::contrib::NmsCUDAKernel(
                boxes.data_ptr<float>(), scores.data_ptr<float>(),
                boxes.size(0), nms_overlap_thresh,
                keep_indices.data_ptr<int64_t>());
        // Shrink in-place to the true count; since count <= boxes.size(0)
        // this never reallocates, so it's metadata-only (no extra copy).
        return keep_indices.resize_({count});
#else
        TORCH_CHECK(false, "Nms was not compiled with CUDA support")

#endif
    } else if (boxes.is_xpu()) {
#ifdef BUILD_SYCL_MODULE
        sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
        torch::Tensor keep_indices = torch::empty(
                {boxes.size(0)},
                torch::TensorOptions().dtype(torch::kLong).device(
                        boxes.device()));
        int count = open3d::ml::contrib::NmsSYCLKernel(
                queue, boxes.data_ptr<float>(), scores.data_ptr<float>(),
                boxes.size(0), nms_overlap_thresh,
                keep_indices.data_ptr<int64_t>());
        // Shrink in-place to the true count; since count <= boxes.size(0)
        // this never reallocates, so it's metadata-only (no extra copy).
        return keep_indices.resize_({count});
#else
        TORCH_CHECK(false, "Nms was not compiled with SYCL support")

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
