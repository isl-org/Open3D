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

#include "open3d/core/hashmap/TensorHashmap.h"

#include <unordered_map>

namespace open3d {
namespace core {
std::pair<Tensor, Tensor> TensorHashmap::Unique(const Tensor &tensor) {
    std::vector<int64_t> indices_data(tensor.GetShape()[0]);
    std::iota(indices_data.begin(), indices_data.end(), 0);
    Tensor indices(indices_data, {tensor.GetShape()[0]}, Dtype::Int64,
                   tensor.GetDevice());
    auto tensor_hash = std::make_shared<TensorHashmap>(tensor, indices, false);
    return tensor_hash->Insert(tensor, indices);
}

TensorHashmap::TensorHashmap(const Tensor &coords,
                             const Tensor &values,
                             bool insert /* = true */) {
    // Device check
    if (coords.GetDevice().GetType() != values.GetDevice().GetType()) {
        utility::LogError("TensorHashmap::Input tensors device mismatch.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("TensorHashmap::Input tensors must be contiguous.");
    }

    // Shape check
    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() != 2) {
        utility::LogError("TensorHashmap::Input coords shape must be (N, D).");
    }
    if (coords_shape[0] != values_shape[0]) {
        utility::LogError(
                "TensorHashmap::Input coords and values size mismatch.");
    }

    // Store type and dim info
    key_dtype_ = coords.GetDtype();
    val_dtype_ = values.GetDtype();
    key_dim_ = coords_shape[1];
    val_dim_ = values_shape.size() == 1 ? 1 : values_shape[1];

    int64_t N = coords_shape[0];

    size_t key_size = key_dtype_.ByteSize() * key_dim_;
    if (key_size > MAX_KEY_BYTESIZE) {
        utility::LogError(
                "TensorHashmap::Unsupported key size: at most {} bytes per "
                "key is "
                "supported, received {} bytes per key",
                MAX_KEY_BYTESIZE, key_size);
    }
    size_t value_size = val_dtype_.ByteSize() * val_dim_;

    Dtype key_dtype(Dtype::DtypeCode::Object, key_size, "key");
    Dtype val_dtype(Dtype::DtypeCode::Object, value_size, "val");

    // Create hashmap and reserve twice input size
    hashmap_ = std::make_shared<Hashmap>(N, key_dtype, val_dtype,
                                         coords.GetDevice());

    if (insert) {
        auto iterators = MemoryManager::Malloc(sizeof(iterator_t) * N,
                                               coords.GetDevice());
        auto masks =
                MemoryManager::Malloc(sizeof(bool) * N, coords.GetDevice());

        hashmap_->Insert(static_cast<void *>(coords.GetBlob()->GetDataPtr()),
                         static_cast<void *>(values.GetBlob()->GetDataPtr()),
                         static_cast<iterator_t *>(iterators),
                         static_cast<bool *>(masks), N);
        MemoryManager::Free(iterators, coords.GetDevice());
        MemoryManager::Free(masks, coords.GetDevice());
    }
}

std::pair<Tensor, Tensor> TensorHashmap::Insert(const Tensor &coords,
                                                const Tensor &values) {
    // Device check
    if (coords.GetDevice().GetType() != hashmap_->GetDevice().GetType()) {
        utility::LogError(
                "TensorHashmap::Input tensors and hashmap device mismatch.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("TensorHashmap::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_dtype_ != coords.GetDtype() || val_dtype_ != values.GetDtype()) {
        utility::LogError("TensorHashmap::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("TensorHashmap::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("TensorHashmap::Input coords shape mismatch.");
    }
    auto val_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (val_dim != val_dim_) {
        utility::LogError("TensorHashmap::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Insert
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_coord_tensor(SizeVector({N, key_dim_}), key_dtype_,
                               coords.GetDevice());
    Tensor output_mask_tensor(SizeVector({N}), Dtype::Bool, coords.GetDevice());
    hashmap_->Insert(
            static_cast<uint8_t *>(coords.GetBlob()->GetDataPtr()),
            static_cast<uint8_t *>(values.GetBlob()->GetDataPtr()),
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()), N);
    hashmap_->UnpackIterators(
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()),
            static_cast<void *>(output_coord_tensor.GetBlob()->GetDataPtr()),
            /* value = */ nullptr, N);

    MemoryManager::Free(iterators, coords.GetDevice());
    return std::make_pair(output_coord_tensor, output_mask_tensor);
}

std::pair<Tensor, Tensor> TensorHashmap::Find(const Tensor &coords) {
    // Device check
    if (coords.GetDevice().GetType() != hashmap_->GetDevice().GetType()) {
        utility::LogError(
                "TensorHashmap::Input tensors and hashmap device mismatch.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous()) {
        utility::LogError("TensorHashmap::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_dtype_ != coords.GetDtype()) {
        utility::LogError("TensorHashmap::Input coords key type mismatch.");
    }
    auto coords_shape = coords.GetShape();
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("TensorHashmap::Input coords shape mismatch.");
    }
    int64_t N = coords.GetShape()[0];

    // Search
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_value_tensor(SizeVector({N, val_dim_}), val_dtype_,
                               coords.GetDevice());
    Tensor output_mask_tensor(SizeVector({N}), Dtype::Bool, coords.GetDevice());

    hashmap_->Find(
            static_cast<uint8_t *>(coords.GetBlob()->GetDataPtr()),
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()), N);

    hashmap_->UnpackIterators(
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()),
            /* coord = */ nullptr,
            static_cast<void *>(output_value_tensor.GetBlob()->GetDataPtr()),
            N);

    MemoryManager::Free(iterators, coords.GetDevice());

    return std::make_pair(output_value_tensor, output_mask_tensor);
}

Tensor TensorHashmap::Assign(const Tensor &coords, const Tensor &values) {
    // Device check
    if (coords.GetDevice().GetType() != hashmap_->GetDevice().GetType()) {
        utility::LogError(
                "TensorHashmap::Input tensors and hashmap device mismatch.");
    }

    // Contiguous check to fit internal hashmap
    if (!coords.IsContiguous() || !values.IsContiguous()) {
        utility::LogError("TensorHashmap::Input tensors must be contiguous.");
    }

    // Type and shape check
    if (key_dtype_ != coords.GetDtype() || val_dtype_ != values.GetDtype()) {
        utility::LogError("TensorHashmap::Input key/value type mismatch.");
    }

    auto coords_shape = coords.GetShape();
    auto values_shape = values.GetShape();
    if (coords_shape.size() == 0 || coords_shape.size() == 0) {
        utility::LogError("TensorHashmap::Inputs are empty tensors");
    }
    if (coords_shape.size() != 2 || coords_shape[1] != key_dim_) {
        utility::LogError("TensorHashmap::Input coords shape mismatch.");
    }
    auto val_dim = values_shape.size() == 1 ? 1 : values_shape[1];
    if (val_dim != val_dim_) {
        utility::LogError("TensorHashmap::Input values shape mismatch.");
    }

    int64_t N = coords.GetShape()[0];

    // Search
    auto iterators =
            MemoryManager::Malloc(sizeof(iterator_t) * N, coords.GetDevice());

    Tensor output_mask_tensor(SizeVector({N}), Dtype::UInt8,
                              coords.GetDevice());

    hashmap_->Find(
            static_cast<uint8_t *>(coords.GetBlob()->GetDataPtr()),
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()), N);

    hashmap_->AssignIterators(
            static_cast<iterator_t *>(iterators),
            static_cast<bool *>(output_mask_tensor.GetBlob()->GetDataPtr()),
            static_cast<uint8_t *>(values.GetBlob()->GetDataPtr()), N);

    MemoryManager::Free(iterators, coords.GetDevice());

    return output_mask_tensor;
}
}  // namespace core
}  // namespace open3d
