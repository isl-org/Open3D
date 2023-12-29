// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/TensorFunction.h"

#include "open3d/core/kernel/Kernel.h"

namespace open3d {
namespace core {

static Tensor ConcatenateImpl(const std::vector<Tensor>& tensors,
                              const int64_t axis) {
    const int num_tensors = tensors.size();
    const int64_t num_dims = tensors[0].NumDims();
    const int64_t axis_d = shape_util::WrapDim(axis, num_dims);

    const Device device = tensors[0].GetDevice();
    const Dtype dtype = tensors[0].GetDtype();
    SizeVector combined_shape = tensors[0].GetShape();

    // Asserts input tensor properties such as device, dtype and dimensions.
    for (int i = 1; i < num_tensors; ++i) {
        core::AssertTensorDevice(tensors[i], device);
        core::AssertTensorDtype(tensors[i], dtype);

        if (tensors[i].NumDims() != num_dims) {
            utility::LogError(
                    "All the input tensors must have same number of "
                    "dimensions, but the tensor at index 0 has {} dimension(s) "
                    "and the tensor at index {} has {} dimension(s).",
                    num_dims, i, tensors[i].NumDims());
        }

        // Check Shape.
        for (int64_t j = 0; j < num_dims; ++j) {
            if (j != axis_d && combined_shape[j] != tensors[i].GetShape(j)) {
                utility::LogError(
                        "All the input tensor dimensions, other than dimension "
                        "size along concatenation axis must be same, but along "
                        "dimension {}, the tensor at index 0 has size {} and "
                        "the tensor at index {} has size {}.",
                        j, combined_shape[j], i, tensors[i].GetShape(j));
            }
        }

        combined_shape[axis_d] += tensors[i].GetShape(axis_d);
    }

    // Common TensorKey for dimensions < axis_d.
    std::vector<TensorKey> common_tks;
    for (int i = 0; i < axis_d; ++i) {
        common_tks.push_back(TensorKey::Slice(0, combined_shape[i], 1));
    }

    Tensor combined_tensor(combined_shape, dtype, device);

    // Cumulate length along `axis`.
    int64_t cumulated_length = 0;
    for (int i = 0; i < num_tensors; ++i) {
        const int64_t local_length = tensors[i].GetShape(axis_d);

        // TensorKey(s) for individual tensors.
        std::vector<TensorKey> local_tks = common_tks;
        local_tks.push_back(TensorKey::Slice(
                cumulated_length, cumulated_length + local_length, 1));

        cumulated_length += local_length;

        combined_tensor.SetItem(local_tks, tensors[i]);
    }

    return combined_tensor;
}

Tensor Concatenate(const std::vector<Tensor>& tensors,
                   const utility::optional<int64_t>& axis) {
    const int num_tensors = tensors.size();

    if (num_tensors < 1) {
        utility::LogError("Expected at least 1 tensor, but got 0.");
    }
    if (num_tensors == 1) {
        std::vector<Tensor> split_tensors;
        split_tensors.reserve(tensors[0].GetLength());

        for (int i = 0; i < tensors[0].GetLength(); ++i) {
            split_tensors.push_back(tensors[0][i]);
        }

        return Concatenate(split_tensors, axis);
    }

    if (!axis.has_value()) {
        std::vector<Tensor> flattened_tensors;
        for (int i = 0; i < num_tensors; ++i) {
            // TODO: Implement Tensor::FlattenTensor
            flattened_tensors.push_back(
                    tensors[i].Reshape({tensors[i].NumElements(), 1}));
        }

        return ConcatenateImpl(flattened_tensors, 0).Reshape({-1});
    } else {
        if (tensors[0].NumDims() == 0) {
            utility::LogError(
                    "Zero-dimensional tensor can only be concatenated along "
                    "axis = null, but got {}.",
                    axis.value());
        }

        return ConcatenateImpl(tensors, axis.value());
    }
}

Tensor Append(const Tensor& self,
              const Tensor& other,
              const utility::optional<int64_t>& axis) {
    return Concatenate({self, other}, axis);
}

Tensor Maximum(const Tensor& input, const Tensor& other) {
    core::AssertTensorDevice(input, other.GetDevice());
    core::AssertTensorDtype(input, other.GetDtype());

    Tensor dst_tensor(
            shape_util::BroadcastedShape(input.GetShape(), other.GetShape()),
            input.GetDtype(), input.GetDevice());
    kernel::BinaryEW(input, other, dst_tensor, kernel::BinaryEWOpCode::Maximum);

    return dst_tensor;
}

Tensor Minimum(const Tensor& input, const Tensor& other) {
    core::AssertTensorDevice(input, other.GetDevice());
    core::AssertTensorDtype(input, other.GetDtype());

    Tensor dst_tensor(
            shape_util::BroadcastedShape(input.GetShape(), other.GetShape()),
            input.GetDtype(), input.GetDevice());
    kernel::BinaryEW(input, other, dst_tensor, kernel::BinaryEWOpCode::Minimum);

    return dst_tensor;
}

}  // namespace core
}  // namespace open3d
