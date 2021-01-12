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

    /// Constructor for primitive and custom types, supporting element shapes.
    /// Example 1:
    /// Key is int<3> coordinate:
    /// # Option 1:
    /// - dtype_key = Dtype::Int32
    /// - element_shape_key = {3}
    /// # Option 2:
    /// - dtype_key = Dtype(DtypeCode::Object, 3 * Dtype::Int32.ByteSize(),
    /// "int3")
    /// - element_shape_key = {1}
    /// Example 2:
    /// Key is struct Pt {int x; int y; int z;}
    /// - dtype_key = Dtype(DtypeCode::Object, sizeof(Pt), "pt")
    /// - element_shape_key = {1}
    Hashmap(int64_t init_capacity,
            const Dtype& dtype_key,
            const Dtype& dtype_value,
            const SizeVector& element_shape_key,
            const SizeVector& element_shape_value,
            const Device& device);

    ~Hashmap(){};

    /// Rehash expects extra memory space at runtime, since it consists of
    /// 1) dumping all key value pairs to a buffer
    /// 2) deallocate old hash table
    /// 3) create a new hash table
    /// 4) parallel insert dumped key value pairs
    void Rehash(int64_t buckets);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_addrs,
                Tensor& output_masks);

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

    /// Parallel find an array of keys in Tensor.
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    /// \masks: success insertions, must be combined with \addrs in advanced
    /// indexing.
    void Find(const Tensor& input_keys,
              Tensor& output_addrs,
              Tensor& output_masks);

    /// Parallel erase an array of keys in Tensor.
    /// Output masks is a bool Tensor.
    /// Return \masks: success insertions, must be combined with \addrs in
    /// advanced indexing.
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Parallel collect all iterators in the hash table
    /// Return \addrs: internal indices that can be directly used for advanced
    /// indexing in Tensor key/value buffers.
    void GetActiveIndices(Tensor& output_indices) const;

    Hashmap Clone() const;
    Hashmap To(const Device& device, bool copy = false) const;
    Hashmap CPU() const;
    Hashmap CUDA(int device_id = 0) const;

    int64_t Size() const;

    int64_t GetCapacity() const;
    int64_t GetBucketCount() const;
    Device GetDevice() const;
    int64_t GetKeyBytesize() const;
    int64_t GetValueBytesize() const;

    Tensor& GetKeyBuffer() const;
    Tensor& GetValueBuffer() const;

    Tensor GetKeyTensor() const;
    Tensor GetValueTensor() const;

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

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

    SizeVector element_shape_key_;
    SizeVector element_shape_value_;
};

}  // namespace core
}  // namespace open3d
