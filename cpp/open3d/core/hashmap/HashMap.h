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

namespace open3d {
namespace core {

class DeviceHashBackend;

enum class HashBackendType { Slab, StdGPU, TBB, Default };

/// Tensor hash map (unique keys) on CPU, CUDA, or SYCL.
///
/// \section HashMapOverview Overview
///
/// Keys and values are stored in internal **buffers**. \ref Insert and \ref Find
/// return per-input-row **buf_indices** and **masks**. Use buf_indices with
/// \ref GetKeyTensor() and \ref GetValueTensors() (advanced indexing). Backend
/// implementations differ; behavior below is the portable contract.
///
/// \section HashMapBufIndices Buffer indices (important for correct use)
///
/// | Property | Detail |
/// |----------|--------|
/// | Type | `Int32` in return tensors — cast to `Int64` before \ref Tensor::IndexGet |
/// | Meaning | Valid **gather index** into key/value buffers for that input row |
/// | Not guaranteed | Contiguous range `[0, Size())` across all returned indices |
/// | SYCL caveat | Duplicate-key insert races can leave **holes** in the buffer (unused slots freed by the backend) |
/// | CPU/CUDA | Often dense after pure inserts, but **do not rely** on density |
///
/// **Recommended patterns**
/// - List all live entries: \ref GetActiveIndices() (length = \ref Size()).
/// - First occurrence per key on insert: combine \c masks (true = new key).
/// - Dense row ids `[0, num_unique)`: remap from unique insert slots (see
///   `t::geometry::PointCloud::VoxelDownSample` on SYCL).
///
/// Example (keys for successful inserts only):
/// \code
/// auto [buf_i, mask] = map.Insert(keys, values);
/// Tensor active_keys = map.GetKeyTensor().IndexGet(
///         {buf_i.To(Int64).IndexGet({mask})});
/// \endcode
///
/// \section HashMapBackends Backends (device-dependent)
///
/// | Device | Default backend |
/// |--------|-----------------|
/// | CPU | TBB |
/// | CUDA | stdgpu (slab optional) |
/// | SYCL | \ref SYCLHashBackend (see `SYCL/SYCLHashBackend.h`) |
///
/// Growth: call \ref Reserve to rehash when load factor requires it.
class HashMap : public IsDevice {
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

    /// Reserve the internal hash map with the given capacity by rehashing.
    void Reserve(int64_t capacity);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Return: \p output_buf_indices — buffer gather indices (see class docs;
    /// convert Int32 → Int64 for indexing). \p output_masks — true if this row
    /// inserted a new key; false if the key was already present.
    std::pair<Tensor, Tensor> Insert(const Tensor& input_keys,
                                     const Tensor& input_values);

    /// Parallel insert arrays of keys and a structure of value arrays in
    /// Tensors.
    /// Return: output_buf_indices and output_masks, their role are the same as
    /// in single value Insert interface.
    std::pair<Tensor, Tensor> Insert(
            const Tensor& input_keys,
            const std::vector<Tensor>& input_values_soa);

    /// Parallel activate arrays of keys in Tensor.
    /// Specifically useful for large value elements (e.g., a 3D tensor), where
    /// we can do in-place management after activation.
    /// Return: output_buf_indices and output_masks, their roles are the same
    /// as in Insert.
    std::pair<Tensor, Tensor> Activate(const Tensor& input_keys);

    /// Parallel find an array of keys in Tensor.
    /// Return: \p output_buf_indices and \p output_masks (same semantics as
    /// \ref Insert); false mask if key not found.
    std::pair<Tensor, Tensor> Find(const Tensor& input_keys);

    /// Parallel erase an array of keys in Tensor.
    /// Return: output_masks stores if the erase is a success or failure (key
    /// not found all already erased in another thread).
    Tensor Erase(const Tensor& input_keys);

    /// Collect buffer indices for all active entries (length \ref Size()).
    /// Indices are valid for \ref GetKeyTensor() / value buffers but may still
    /// be non-contiguous in the underlying buffer on SYCL.
    Tensor GetActiveIndices() const;

    /// Same as Insert with a single value array, but takes output_buf_indices
    /// and output_masks as input. If their shapes and types match, reallocation
    /// is not needed.
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Same as Insert with a SoA of values, but takes output_buf_indices
    /// and output_masks as input. If their shapes and types match, reallocation
    /// is not needed.
    void Insert(const Tensor& input_keys,
                const std::vector<Tensor>& input_values_soa,
                Tensor& output_buf_indices,
                Tensor& output_masks);

    /// Same as Activate, but takes output_buf_indices
    /// and output_masks as input. If their shapes and types match, reallocation
    /// is not needed.
    void Activate(const Tensor& input_keys,
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
    Device GetDevice() const override;

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
                    Tensor& output_masks,
                    bool is_activate_op = false);

    void CheckKeyLength(const Tensor& input_keys) const;
    void CheckKeyValueLengthCompatibility(
            const Tensor& input_keys,
            const std::vector<Tensor>& input_values_soa) const;
    void CheckKeyCompatibility(const Tensor& input_keys) const;
    void CheckValueCompatibility(
            const std::vector<Tensor>& input_values_soa) const;

    void PrepareIndicesOutput(Tensor& output_buf_indices, int64_t length) const;
    void PrepareMasksOutput(Tensor& output_masks, int64_t length) const;

    std::pair<int64_t, std::vector<int64_t>> GetCommonValueSizeDivisor();

private:
    std::shared_ptr<DeviceHashBackend> device_hashmap_;

    Dtype key_dtype_;
    SizeVector key_element_shape_;

    std::vector<Dtype> dtypes_value_;
    std::vector<SizeVector> element_shapes_value_;
};

}  // namespace core
}  // namespace open3d
