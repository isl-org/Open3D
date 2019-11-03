// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Indexing.h"
#include "Tensor.h"

namespace open3d {
std::pair<std::vector<Tensor>, SizeVector> PreprocessIndexTensors(
        const Tensor& tensor, const std::vector<Tensor>& indices) {
    std::vector<Tensor> output_indices;
    SizeVector output_shape;

    size_t non_trivial_index_size = 0;
    const auto& tensor_shape = tensor.GetShape();
    size_t i = 0;
    for (; i < indices.size(); ++i) {
        const Tensor& index = indices[i];
        const auto& index_shape = index.GetShape();
        if (!index.IsContiguous()) {
            utility::LogError("Only contiguous indexing tensors are supported");
        }
        if (index.GetDtype() != Dtype::Int32) {
            utility::LogError("Only Int32 indexing tensors are supported");
        }
        if (index_shape.size() > 1) {
            utility::LogError("Only 1D indexing tensors are supported");
        }

        /// All elements (no element)
        if (index_shape.size() == 0 || index_shape[0] == 0) {
            output_shape.emplace_back(tensor_shape[i]);
        }

        /// Broadcasting (one element)
        if (index_shape[0] == 1) {
            output_shape.emplace_back(1);
        } else {
            if (non_trivial_index_size != 0 &&
                non_trivial_index_size != index_shape[i]) {
                utility::LogError("Index shapes mismatch on dim {}: {} vs {}",
                                  i, non_trivial_index_size, index_shape[i]);
            }
            if (non_trivial_index_size == 0) {
                non_trivial_index_size = index_shape[i];
            }
            output_shape.emplace_back(non_trivial_index_size);
        }

        output_indices.emplace_back(index);
    }

    Tensor empty_index = Tensor(SizeVector(), Dtype::Int32, tensor.GetDevice());
    for (; i < tensor_shape.size(); ++i) {
        output_shape.emplace_back(tensor_shape[i]);
        output_indices.emplace_back(empty_index);
    }

    return std::make_pair(output_indices, output_shape);
}

}  // namespace open3d
