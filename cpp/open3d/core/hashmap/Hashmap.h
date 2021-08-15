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
#include "open3d/core/hashmap/HashmapBuffer.h"

namespace open3d {
namespace core {

class DeviceHashmap;

enum class HashmapBackend { Slab, StdGPU, TBB, Default };

class Hashmap {
public:
    /// Constructor for primitive types, supporting element shapes.
    /// Example:
    /// Key is int<3> coordinate:
    /// - dtype_key = core::Int32
    /// - element_shape_key = {3}
    Hashmap(int64_t init_capacity,
            const Dtype& dtype_key,
            const SizeVector& element_shape_key,
            const Dtype& dtype_value,
            const SizeVector& element_shape_value,
            const Device& device,
            const HashmapBackend& backend = HashmapBackend::Default);

    Hashmap(int64_t init_capacity,
            const Dtype& dtype_key,
            const SizeVector& element_shape_key,
            const std::vector<Dtype>& dtypes_value,
            const std::vector<SizeVector>& element_shapes_value,
            const Device& device,
            const HashmapBackend& backend = HashmapBackend::Default);

    ~Hashmap(){};

    /// Rehash expects extra memory space at runtime, since it consists of
    /// 1) dumping all key value pairs to a buffer
    /// 2) deallocate old hash table
    /// 3) create a new hash table
    /// 4) parallel insert dumped key value pairs
    void Rehash(int64_t buckets);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Return buf_indices: internal indices that can be directly used for
    /// advanced indexing in Tensor key/value buffers. masks: success
    /// insertions, must be combined with buf_indices in advanced indexing.
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Return buf_indices: internal indices that can be directly used for
    /// advanced indexing in Tensor key/value buffers. masks: success
    /// insertions, must be combined with buf_indices in advanced indexing.
    void Insert(const Tensor& input_keys,
                const std::vector<Tensor>& arr_input_values,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Parallel activate arrays of keys in Tensor.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    /// Return buf_indices: internal indices that can be directly used for
    /// advanced indexing in Tensor key/value buffers. masks: success
    /// insertions, must be combined with buf_indices in advanced indexing.
    void Activate(const Tensor& input_keys,
                  Tensor& output_buf_indices,
                  Tensor& output_masks);

    /// Parallel find an array of keys in Tensor.
    /// Return buf_indices: internal indices that can be directly used for
    /// advanced indexing in Tensor key/value buffers. masks: success
    /// insertions, must be combined with buf_indices in advanced indexing.
    void Find(const Tensor& input_keys,
              Tensor& output_buf_indices,
              Tensor& output_masks);

    /// Parallel erase an array of keys in Tensor.
    /// Output masks is a bool Tensor.
    /// Return masks: success insertions, must be combined with buf_indices in
    /// advanced indexing.
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Parallel collect all iterators in the hash table
    /// Return buf_indices: internal indices that can be directly used for
    /// advanced indexing in Tensor key/value buffers.
    void GetActiveIndices(Tensor& output_buf_indices) const;

    /// Clear stored map without reallocating memory.
    void Clear();

    /// Save active key and value to a npz file at 'key' and 'value'. The file
    /// name should end with npz, otherwise npz will be added as an extension.
    void Save(const std::string& file_name);

    /// Load active key and value from a npz file at 'key' and 'value'. The npz
    /// file should contain a 'key' and a 'value' tensor, of the same length.
    static Hashmap Load(const std::string& file_name);

    Hashmap Clone() const;
    Hashmap To(const Device& device, bool copy = false) const;

    int64_t Size() const;

    int64_t GetCapacity() const;
    int64_t GetBucketCount() const;
    Device GetDevice() const;

    Tensor GetKeyTensor() const;
    std::vector<Tensor> GetValueTensors() const;
    Tensor GetValueTensor(size_t index = 0) const;

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

    std::shared_ptr<DeviceHashmap> GetDeviceHashmap() const {
        return device_hashmap_;
    }

protected:
    void Init(int64_t init_capacity,
              const Device& device,
              const HashmapBackend& backend);

    void InsertImpl(const Tensor& input_keys,
                    const std::vector<Tensor>& arr_input_values,
                    Tensor& output_buf_indices,
                    Tensor& output_masks);

    void CheckKeyLength(const Tensor& input_keys) const;
    void CheckKeyValueLengthCompatibility(
            const Tensor& input_keys,
            const std::vector<Tensor>& arr_input_values) const;
    void CheckKeyCompatibility(const Tensor& input_keys) const;
    void CheckValueCompatibility(
            const std::vector<Tensor>& arr_input_values) const;

    void PrepareIndicesOutput(Tensor& output_buf_indices, int64_t length) const;
    void PrepareMasksOutput(Tensor& output_masks, int64_t length) const;

private:
    std::shared_ptr<DeviceHashmap> device_hashmap_;

    Dtype dtype_key_;
    SizeVector element_shape_key_;

    std::vector<Dtype> dtypes_value_;
    std::vector<SizeVector> element_shapes_value_;
};

}  // namespace core
}  // namespace open3d
