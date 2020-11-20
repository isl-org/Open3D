// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------q

#include <vector>

#include "open3d/ml/impl/misc/Nms.h"
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
        std::vector<int64_t> keep_indices = open3d::ml::impl::NmsCUDAKernel(
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
        std::vector<int64_t> keep_indices = open3d::ml::impl::NmsCPUKernel(
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
