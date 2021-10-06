// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/TensorFunction.h"

namespace open3d {
namespace core {

Tensor Concatenate(const std::vector<Tensor>& tensor_list, int64_t axis) {
    const int num_tensors = tensor_list.size();
    if (num_tensors < 2) {
        utility::LogError("Expected atleast 2 tensors, but got {}.",
                          num_tensors);
    }

    const int64_t num_dims = tensor_list[0].NumDims();
    const int64_t axis_d = shape_util::WrapDim(axis, num_dims);

    const Device device = tensor_list[0].GetDevice();
    const Dtype dtype = tensor_list[0].GetDtype();
    SizeVector combined_shape = tensor_list[0].GetShape();

    // Asserts input tensor properties such as device, dtype and dimentions.
    for (int i = 1; i < num_tensors; ++i) {
        core::AssertTensorDevice(tensor_list[i], device);
        core::AssertTensorDtype(tensor_list[i], dtype);

        if (tensor_list[i].NumDims() != num_dims) {
            utility::LogError(
                    "All the input tensors must have same number of "
                    "dimensions, but the tensor at index 0 has {} dimension(s) "
                    "and the tensor at index {} has {} dimension(s).",
                    num_dims, i, tensor_list[i].NumDims());
        }

        // Check Shape.
        for (int64_t j = 0; j < num_dims; ++j) {
            if (j != axis_d &&
                combined_shape[j] != tensor_list[i].GetShape(j)) {
                utility::LogError(
                        "All the input tensor dimensions, other than dimension "
                        "size along concatenation axis must be same, but along "
                        "dimension {}, the tensor at index 0 has size {} and "
                        "the tensor at index {} has size {}.",
                        j, combined_shape[j], i, tensor_list[i].GetShape(j));
            }
        }

        combined_shape[axis_d] += tensor_list[i].GetShape(axis_d);
    }

    // Common TensorKey for dimensions < axis_d.
    std::vector<TensorKey> common_tks;
    for (int i = 0; i < axis_d; ++i) {
        common_tks.push_back(TensorKey::Slice(0, combined_shape[i], 1));
    }

    Tensor combined_tensor(combined_shape, tensor_list[0].GetDtype(),
                           tensor_list[0].GetDevice());

    // Cumulate length along `axis`.
    int64_t length_cumulator = 0;

    for (int i = 0; i < num_tensors; ++i) {
        const int64_t updated_length =
                length_cumulator + tensor_list[i].GetShape(axis_d);

        // TensorKey(s) for individual tensors.
        std::vector<TensorKey> local_tks = common_tks;
        local_tks.push_back(
                TensorKey::Slice(length_cumulator, updated_length, 1));

        length_cumulator = updated_length;

        combined_tensor.SetItem(local_tks, tensor_list[i]);
    }

    return combined_tensor;
}

Tensor Append(const Tensor& self,
              const Tensor& other,
              const utility::optional<int64_t> axis) {
    if (!axis.has_value()) {
        return Concatenate({self.Reshape({self.NumElements(), 1}),
                            other.Reshape({other.NumElements(), 1})},
                           0)
                .Reshape({-1});
    } else {
        if (self.NumDims() == 0) {
            utility::LogError(
                    "Zero-dimensional tensor can only be appended along axis = "
                    "null, but got {}.",
                    axis.value());
        }

        return Concatenate({self, other}, axis.value());
    }
}

}  // namespace core
}  // namespace open3d
