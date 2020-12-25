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
// ----------------------------------------------------------------------------
//
#pragma once

#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"
#include "torch/script.h"

template <class TReal, class TIndex>
void ContinuousConvCPU(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_importance,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        torch::Tensor& out_features);

#ifdef BUILD_CUDA_MODULE
template <class TReal, class TIndex>
void ContinuousConvCUDA(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_importance,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        torch::Tensor& out_features);
#endif
