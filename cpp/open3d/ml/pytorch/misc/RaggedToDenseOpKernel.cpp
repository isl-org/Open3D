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

#include "open3d/ml/pytorch/misc/RaggedToDenseOpKernel.h"

#include "open3d/ml/impl/misc/RaggedToDense.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
torch::Tensor RaggedToDenseCPU(const torch::Tensor& values,
                               const torch::Tensor& row_splits,
                               const int64_t out_col_size,
                               const torch::Tensor& default_value) {
    auto out_shape = values.sizes().vec();
    out_shape.erase(out_shape.begin());
    out_shape.insert(out_shape.begin(), {row_splits.size(0) - 1, out_col_size});
    torch::Tensor out =
            torch::empty(out_shape, torch::dtype(ToTorchDtype<T>()));

    open3d::ml::impl::RaggedToDenseCPU(
            values.data_ptr<T>(), row_splits.data_ptr<int64_t>(),
            row_splits.size(0), out_col_size, default_value.data_ptr<T>(),
            default_value.numel(), out.data_ptr<T>());

    return out;
}

#define INSTANTIATE(T)                                                 \
    template torch::Tensor RaggedToDenseCPU<T>(                        \
            const torch::Tensor&, const torch::Tensor&, const int64_t, \
            const torch::Tensor&);

INSTANTIATE(uint8_t)
INSTANTIATE(int8_t)
INSTANTIATE(int16_t)
INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
INSTANTIATE(float)
INSTANTIATE(double)
