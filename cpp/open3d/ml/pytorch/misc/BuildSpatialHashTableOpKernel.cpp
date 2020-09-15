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

#include "open3d/ml/impl/misc/FixedRadiusSearch.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
void BuildSpatialHashTableCPU(const torch::Tensor& points,
                              double radius,
                              const torch::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              torch::Tensor& hash_table_index,
                              torch::Tensor& hash_table_cell_splits) {
    open3d::ml::impl::BuildSpatialHashTableCPU(
            points.size(0), points.data_ptr<T>(), T(radius),
            points_row_splits.size(0), points_row_splits.data_ptr<int64_t>(),
            hash_table_splits.data(), hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());
}
#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableCPU<T>(                  \
            const torch::Tensor&, double, const torch::Tensor&, \
            const std::vector<uint32_t>&, torch::Tensor&, torch::Tensor&);

INSTANTIATE(float)
INSTANTIATE(double)
