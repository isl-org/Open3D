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
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {

class Hashmap {
public:
    static constexpr uint32_t kDefaultElemsPerBucket = 4;

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
    /// Output iterators and masks can be nullptrs if return iterators are not
    /// to be processed.
    void Insert(const void* input_keys,
                const void* input_values,
                iterator_t* output_iterators,
                bool* output_masks,
                int64_t count);

    /// Parallel insert arrays of keys and values in Tensors.
    /// Output iterators and masks are Tensors and can be further processed
    void Insert(const Tensor& input_keys,
                const Tensor& input_values,
                Tensor& output_iterators,
                Tensor& output_masks);

    /// Parallel activate arrays of keys without copying values.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    void Activate(const void* input_keys,
                  iterator_t* output_iterators,
                  bool* output_masks,
                  int64_t count);

    /// Parallel activate arrays of keys in Tensor.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    void Activate(const Tensor& input_keys,
                  Tensor& output_iterators,
                  Tensor& output_masks);

    /// Parallel find an array of keys.
    /// Output iterators and masks CANNOT be nullptrs as we have to interpret
    /// them.
    void Find(const void* input_keys,
              iterator_t* output_iterators,
              bool* output_masks,
              int64_t count);

    /// Parallel find an array of keys in Tensor.
    /// Output iterators is an object Tensor, masks is a bool Tensor.
    void Find(const Tensor& input_keys,
              Tensor& output_iterators,
              Tensor& output_masks);

    /// Parallel erase an array of keys.
    /// Output masks can be a nullptr if return results are not to be
    /// processed.
    void Erase(const void* input_keys, bool* output_masks, int64_t count);

    /// Parallel erase an array of keys in Tensor.
    /// Output masks is a bool Tensor.
    void Erase(const Tensor& input_keys, Tensor& output_masks);

    /// Parallel collect all iterators in the hash table
    int64_t GetIterators(iterator_t* output_iterators);

    /// Parallel unpack iterators to contiguous arrays of keys and/or
    /// values. Output keys and values can be nullptrs if they are not
    /// to be processed/stored.
    void UnpackIterators(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         int64_t count);

    /// Parallel assign iterators in-place with associated values.
    /// Note: users should manage the key-value correspondences around
    /// iterators.
    void AssignIterators(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         int64_t count);

    int64_t Size() const;
    int64_t GetCapacity() const { return device_hashmap_->GetCapacity(); }

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    std::vector<int64_t> BucketSizes() const;

    /// Return size / bucket_count.
    float LoadFactor() const;

    void AssertKeyDtype(const Dtype& dtype_key) const;
    void AssertValueDtype(const Dtype& dtype_val) const;

    Dtype GetKeyDtype() const { return dtype_key_; }
    Dtype GetValueDtype() const { return dtype_val_; }

    Device GetDevice() const { return device_hashmap_->GetDevice(); }

    Tensor GetKeyBlobAsTensor(const SizeVector& shape, Dtype dtype) {
        return device_hashmap_->GetKeyBlobAsTensor(shape, dtype);
    }

    Tensor GetValueBlobAsTensor(const SizeVector& shape, Dtype dtype) {
        return device_hashmap_->GetValueBlobAsTensor(shape, dtype);
    }

private:
    std::shared_ptr<DefaultDeviceHashmap> device_hashmap_;

    Dtype dtype_key_ = Dtype::Undefined;
    Dtype dtype_val_ = Dtype::Undefined;
};

}  // namespace core
}  // namespace open3d
