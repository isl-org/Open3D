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

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashmapBuffer.h"

namespace open3d {
namespace core {

// Forward declaration of device-dependent classes
class DefaultHash;
class DefaultKeyEq;

template <typename Hash, typename KeyEq>
class DeviceHashmap;
typedef DeviceHashmap<DefaultHash, DefaultKeyEq> DefaultDeviceHashmap;

class Hashmap {
public:
    static constexpr int64_t kDefaultElemsPerBucket = 4;

    // Default constructor for common users.
    Hashmap(int64_t init_capacity,
            Dtype dtype_key,
            Dtype dtype_val,
            const Device& device);

    ~Hashmap(){};

    /// Rehash expects extra memory space at runtime, since it consists of
    /// 1) dumping all key value pairs to a buffer
    /// 2) deallocate old hash table
    /// 3) create a new hash table
    /// 4) parallel insert dumped key value pairs
    void Rehash(int64_t buckets);

    /// Parallel insert arrays of keys and values.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Insert(const void* input_keys,
                const void* input_values,
                addr_t* output_addrs,
                bool* output_masks,
                int64_t count);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_addrs,
                Tensor& output_masks);

    /// Parallel activate arrays of keys without copying values.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Activate(const void* input_keys,
                  addr_t* output_addrs,
                  bool* output_masks,
                  int64_t count);

    /// Parallel activate arrays of keys in Tensor.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Activate(const Tensor& input_keys,
                  Tensor& output_addrs,
                  Tensor& output_masks);

    /// Parallel find an array of keys.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Find(const void* input_keys,
              addr_t* output_addrs,
              bool* output_masks,
              int64_t count);

    /// Parallel find an array of keys in Tensor.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Find(const Tensor& input_keys,
              Tensor& output_addrs,
              Tensor& output_masks);

    /// Parallel erase an array of keys.
    /// Output masks can be a nullptr if return results are not to be
    /// processed.
    /// Return \masks: success insertions, must be combined with \addrs in
    /// advanced indexing.
    void Erase(const void* input_keys, bool* output_masks, int64_t count);

    /// Parallel erase an array of keys in Tensor.
    /// Output masks is a bool Tensor.
    /// Return \masks: success insertions, must be combined with \addrs in
    /// advanced indexing.
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Parallel collect all iterators in the hash table
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    int64_t GetActiveIndices(addr_t* output_indices);

    int64_t Size() const;

    int64_t GetCapacity() const;
    int64_t GetBucketCount() const;
    Device GetDevice() const;
    int64_t GetKeyBytesize() const;
    int64_t GetValueBytesize() const;

    Tensor& GetKeyTensor();
    Tensor& GetValueTensor();

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

    /// Helper to access buffer.
    /// Example usage:
    /// For (N, 8, 8, 8) tensors as keys, hashmap will convert them to
    /// (N, 256) arrays, where key size is 256*sizeof(dtype) and key type as
    /// (Object, 256*dsize, "_hash_k"). To access buffer and access via their
    /// original type and shape, reinterpret is required.
    static Tensor ReinterpretBufferTensor(Tensor& buffer,
                                          const SizeVector& shape,
                                          Dtype dtype) {
        if (dtype.ByteSize() * shape.NumElements() !=
            buffer.GetDtype().ByteSize() * buffer.GetShape().NumElements()) {
            utility::LogError(
                    "[Hashmap] Reinterpret buffer as Tensor expects same byte "
                    "size");
        }
        return Tensor(shape, Tensor::DefaultStrides(shape), buffer.GetDataPtr(),
                      dtype, buffer.GetBlob());
    }

protected:
    void AssertKeyDtype(const Dtype& dtype_key,
                        const SizeVector& elem_shape) const;
    void AssertValueDtype(const Dtype& dtype_val,
                          const SizeVector& elem_shape) const;

    Dtype GetKeyDtype() const { return dtype_key_; }
    Dtype GetValueDtype() const { return dtype_value_; }

private:
    std::shared_ptr<DefaultDeviceHashmap> device_hashmap_;

    Dtype dtype_key_ = Dtype::Undefined;
    Dtype dtype_value_ = Dtype::Undefined;
};

}  // namespace core
}  // namespace open3d
