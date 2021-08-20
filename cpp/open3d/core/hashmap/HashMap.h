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

#pragma once

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"

namespace open3d {
namespace core {

class DeviceHashBackend;

enum class HashBackendType { Slab, StdGPU, TBB, Default };

class HashMap {
public:
    /// Initialize a hash map given a key and a value dtype and element shape.
    HashMap(int64_t init_capacity,
            const Dtype& key_dtype,
            const SizeVector& key_element_shape,
            const Dtype& value_dtype,
            const SizeVector& value_element_shapes,
            const Device& device,
            const HashBackendType& backend = HashBackendType::Default);

    /// Initialize a hash map given a key dtype and element shape, and a vector
    /// of value dtypes and element shapes for values stored in structure of
    /// arrays.
    HashMap(int64_t init_capacity,
            const Dtype& key_dtype,
            const SizeVector& key_element_shape,
            const std::vector<Dtype>& dtypes_value,
            const std::vector<SizeVector>& element_shapes_value,
            const Device& device,
            const HashBackendType& backend = HashBackendType::Default);

    /// Default destructor.
    ~HashMap() = default;

    /// Rehash the internal hash map with the given number of buckets.
    void Rehash(int64_t buckets);

    /// Parallel insert arrays of keys and values in Tensors.
    /// The output_buf_indices and output_masks will be overwritten, to be used
    /// with tensor buffers.
    /// output_buf_indices stores buffer indices that access buffer tensors
    /// obtained from GetKeyTensor() and GetValueTensor() via advanced indexing.
    /// NOTE: output_buf_indices are stored in Int32. A conversion to Int64 is
    /// required for further indexing.
    /// output_masks stores if the insertion is
    /// a success or failure (key already exists).
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Parallel insert arrays of keys and a structure of value arrays in
    /// Tensors.
    /// The roles of output_buf_indices and output_masks are the same as Insert
    /// with a single input_value tensor input.
    void Insert(const Tensor& input_keys,
                const std::vector<Tensor>& input_values_soa,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Parallel activate arrays of keys in Tensor.
    /// Specifically useful for large value elements (e.g., a 3D tensor), where
    /// we can do in-place management after activation.
    /// The roles of output_buf_indices and output_masks are the same as Insert.
    void Activate(const Tensor& input_keys,
                  Tensor& output_buf_indices,
                  Tensor& output_masks);

    /// Parallel find an array of keys in Tensor.
    /// The roles of output_buf_indices is the same as Insert.
    /// output_masks stores if the finding is a success or failure (key
    /// not found).
    void Find(const Tensor& input_keys,
              Tensor& output_buf_indices,
              Tensor& output_masks);

    /// Parallel erase an array of keys in Tensor.
    /// The output_masks will be overwritten, to be used
    /// with tensor buffers.
    /// output_masks stores if the erase is a success or failure (key not found
    /// all already erased in another thread).
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Parallel collect all indices in the buffer corresponding to the active
    /// entries in the hash map. Stored in output_buf_indices.
    void GetActiveIndices(Tensor& output_buf_indices) const;

    /// Clear stored map without reallocating the buffers.
    void Clear();

    /// Save active keys and values to a npz file at 'key' and 'value_{:03d}'.
    /// The number of values is stored in 'n_values'.
    /// The file name should end with 'npz', otherwise 'npz' will be added as an
    /// extension.
    void Save(const std::string& file_name);

    /// Load active keys and values from a npz file that contains 'key',
    /// 'n_values', 'value_{:03d}'.
    static HashMap Load(const std::string& file_name);

    /// Clone the hash map with buffers.
    HashMap Clone() const;

    /// Convert the hash map to another device.
    HashMap To(const Device& device, bool copy = false) const;

    /// Get the size (number of active entries) of the hash map.
    int64_t Size() const;

    /// Get the capacity of the hash map.
    int64_t GetCapacity() const;

    /// Get the number of buckets of the internal hash map.
    int64_t GetBucketCount() const;

    /// Get the device of the hash map.
    Device GetDevice() const;

    /// Get the key tensor buffer to be used along with buf_indices and masks.
    /// Example:
    /// GetKeyTensor().IndexGet({buf_indices.To(core::Int64).IndexGet{masks}})
    Tensor GetKeyTensor() const;

    /// Get the values tensor buffers to be used along with buf_indices and
    /// masks. Example:
    /// GetValueTensors()[0].IndexGet({buf_indices.To(core::Int64).IndexGet{masks}})
    std::vector<Tensor> GetValueTensors() const;

    /// Get the i-th value tensor buffer to be used along with buf_indices and
    /// masks. Example:
    /// GetValueTensors(0).IndexGet({buf_indices.To(core::Int64).IndexGet{masks}})
    Tensor GetValueTensor(size_t index = 0) const;

    /// Return number of elements per bucket.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

    /// Return the implementation of the device hash backend.
    std::shared_ptr<DeviceHashBackend> GetDeviceHashBackend() const {
        return device_hashmap_;
    }

protected:
    void Init(int64_t init_capacity,
              const Device& device,
              const HashBackendType& backend);

    void InsertImpl(const Tensor& input_keys,
                    const std::vector<Tensor>& input_values_soa,
                    Tensor& output_buf_indices,
                    Tensor& output_masks);

    void CheckKeyLength(const Tensor& input_keys) const;
    void CheckKeyValueLengthCompatibility(
            const Tensor& input_keys,
            const std::vector<Tensor>& input_values_soa) const;
    void CheckKeyCompatibility(const Tensor& input_keys) const;
    void CheckValueCompatibility(
            const std::vector<Tensor>& input_values_soa) const;

    void PrepareIndicesOutput(Tensor& output_buf_indices, int64_t length) const;
    void PrepareMasksOutput(Tensor& output_masks, int64_t length) const;

private:
    std::shared_ptr<DeviceHashBackend> device_hashmap_;

    Dtype key_dtype_;
    SizeVector key_element_shape_;

    std::vector<Dtype> dtypes_value_;
    std::vector<SizeVector> element_shapes_value_;
};

}  // namespace core
}  // namespace open3d
