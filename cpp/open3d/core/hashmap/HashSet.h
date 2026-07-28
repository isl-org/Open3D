// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"
#include "open3d/core/hashmap/HashMap.h"

namespace open3d {
namespace core {

/// Tensor hash set (unique keys) on CPU, CUDA, or SYCL.
///
/// Thin wrapper around an internal \ref HashMap (dummy values). **Insert**,
/// **Find**,
/// **GetActiveIndices**, and **buf_indices** semantics are identical to \ref
/// HashMap — see \ref HashMap class documentation for buffer-index rules,
/// Int32→Int64, SYCL buffer holes, and usage examples.
class HashSet : public core::IsDevice {
public:
    /// Initialize a hash set given a key dtype and element shape.
    HashSet(int64_t init_capacity,
            const Dtype& key_dtype,
            const SizeVector& key_element_shape,
            const Device& device,
            const HashBackendType& backend = HashBackendType::Default);

    /// Default destructor.
    ~HashSet() = default;

    /// Reserve the internal hash map with the capcity by rehashing.
    void Reserve(int64_t capacity);

    /// Parallel insert keys. Returns buffer gather indices and per-row masks
    /// (see \ref HashMap class docs for \c buf_indices and Int32 → Int64).
    std::pair<Tensor, Tensor> Insert(const Tensor& input_keys);

    /// Parallel find keys. Same \p output_buf_indices / \p output_masks
    /// semantics as \ref HashMap::Find (see \ref HashMap class docs).
    std::pair<Tensor, Tensor> Find(const Tensor& input_keys);

    /// Parallel erase an array of keys in Tensor.
    /// Return: output_masks stores if the erase is a success or failure (key
    /// not found all already erased in another thread).
    Tensor Erase(const Tensor& input_keys);

    /// Collect buffer indices for all active keys (length \ref Size()).
    Tensor GetActiveIndices() const;

    /// Same as Insert, but takes output_buf_indices
    /// and output_masks as input. If their shapes and types match, reallocation
    /// is not needed.
    void Insert(const Tensor& input_keys,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Same as Find, but takes output_buf_indices
    /// and output_masks as input. If their shapes and types match, reallocation
    /// is not needed.
    void Find(const Tensor& input_keys,
              Tensor& output_buf_indices,
              Tensor& output_masks);

    /// Same as Erase, but takes output_masks as input. If its shape and
    /// type matches, reallocation is not needed.
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Same as GetActiveIndices, but takes output_buf_indices as input. If its
    /// shape and type matches, reallocation is not needed.
    void GetActiveIndices(Tensor& output_buf_indices) const;

    /// Clear stored map without reallocating the buffers.
    void Clear();

    /// Save active keys to a npz file at 'key'.
    /// The file name should end with 'npz', otherwise 'npz' will be added as an
    /// extension.
    void Save(const std::string& file_name);

    /// Load active keys and values from a npz file that contains 'key'.
    static HashSet Load(const std::string& file_name);

    /// Clone the hash set with buffers.
    HashSet Clone() const;

    /// Convert the hash set to another device.
    HashSet To(const Device& device, bool copy = false) const;

    /// Get the size (number of active entries) of the hash set.
    int64_t Size() const;

    /// Get the capacity of the hash set.
    int64_t GetCapacity() const;

    /// Get the number of buckets of the internal hash set.
    int64_t GetBucketCount() const;

    /// Get the device of the hash set.
    Device GetDevice() const override;

    /// Get the key tensor buffer to be used along with buf_indices and masks.
    /// Example:
    /// GetKeyTensor().IndexGet({buf_indices.To(core::Int64).IndexGet{masks}})
    Tensor GetKeyTensor() const;

    /// Return number of elements per bucket.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

    /// Return the implementation of the device hash backend.
    std::shared_ptr<DeviceHashBackend> GetDeviceHashBackend() const;

private:
    HashSet(const HashMap& internal_hashmap);
    std::shared_ptr<HashMap> internal_;
};

}  // namespace core
}  // namespace open3d
