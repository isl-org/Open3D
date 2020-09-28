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

// High level non-templated hashmap interface for basic usages.

// If BUILD_CUDA_MODULE, link DefaultHashmap.cu that contains everything, and
// disable code inside DefaultHashmap.cpp
// Else, link DefaultHashmap.cpp and disregard DefaultHashmap.cu

#include "open3d/core/hashmap/Hashmap.h"

#include <unordered_map>

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {

Hashmap::Hashmap(size_t init_capacity,
                 Dtype dtype_key,
                 Dtype dtype_val,
                 const Device& device)
    : dtype_key_(dtype_key), dtype_val_(dtype_val) {
    device_hashmap_ = CreateDefaultDeviceHashmap(
            std::max(init_capacity / kDefaultElemsPerBucket, size_t(1)),
            init_capacity, dtype_key.ByteSize(), dtype_val.ByteSize(), device);
}

void Hashmap::Rehash(size_t buckets) {
    return device_hashmap_->Rehash(buckets);
}

void Hashmap::Insert(const void* input_keys,
                     const void* input_values,
                     iterator_t* output_iterators,
                     bool* output_masks,
                     size_t count) {
    return device_hashmap_->Insert(input_keys, input_values, output_iterators,
                                   output_masks, count);
}

void Hashmap::Insert(const Tensor& input_keys,
                     const Tensor& input_values,
                     Tensor& output_iterators,
                     Tensor& output_masks) {
    AssertKeyDtype(input_keys.GetDtype());
    AssertValueDtype(input_values.GetDtype());

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible key device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    SizeVector val_shape = input_values.GetShape();
    if (val_shape.size() == 0 || val_shape[0] != shape[0]) {
        utility::LogError("[Hashmap]: Invalid value tensor shape");
    }
    if (input_values.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible value device, expected {}, but got {}",
                GetDevice().ToString(), input_values.GetDevice().ToString());
    }

    int64_t count = shape[0];
    Dtype dtype_it(Dtype::DtypeCode::Object, sizeof(iterator_t), "iterator_t");

    output_iterators = Tensor({count}, dtype_it, GetDevice());
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    Insert(input_keys.GetDataPtr(), input_values.GetDataPtr(),
           static_cast<iterator_t*>(output_iterators.GetDataPtr()),
           static_cast<bool*>(output_masks.GetDataPtr()), count);
}

void Hashmap::Activate(const void* input_keys,
                       iterator_t* output_iterators,
                       bool* output_masks,
                       size_t count) {
    return device_hashmap_->Activate(input_keys, output_iterators, output_masks,
                                     count);
}

void Hashmap::Activate(const Tensor& input_keys,
                       Tensor& output_iterators,
                       Tensor& output_masks) {
    AssertKeyDtype(input_keys.GetDtype());

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];
    Dtype dtype_it(Dtype::DtypeCode::Object, sizeof(iterator_t), "iterator_t");

    output_iterators = Tensor({count}, dtype_it, GetDevice());
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    return Activate(input_keys.GetDataPtr(),
                    static_cast<iterator_t*>(output_iterators.GetDataPtr()),
                    static_cast<bool*>(output_masks.GetDataPtr()), count);
}

void Hashmap::Find(const void* input_keys,
                   iterator_t* output_iterators,
                   bool* output_masks,
                   size_t count) {
    return device_hashmap_->Find(input_keys, output_iterators, output_masks,
                                 count);
}

void Hashmap::Find(const Tensor& input_keys,
                   Tensor& output_iterators,
                   Tensor& output_masks) {
    AssertKeyDtype(input_keys.GetDtype());

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];
    Dtype dtype_it(Dtype::DtypeCode::Object, sizeof(iterator_t), "iterator_t");

    output_masks = Tensor({count}, Dtype::Bool, GetDevice());
    output_iterators = Tensor({count}, dtype_it, GetDevice());

    return Find(input_keys.GetDataPtr(),
                static_cast<iterator_t*>(output_iterators.GetDataPtr()),
                static_cast<bool*>(output_masks.GetDataPtr()), count);
}

void Hashmap::Erase(const void* input_keys, bool* output_masks, size_t count) {
    return device_hashmap_->Erase(input_keys, output_masks, count);
}

void Hashmap::Erase(const Tensor& input_keys, Tensor& output_masks) {
    AssertKeyDtype(input_keys.GetDtype());

    SizeVector shape = input_keys.GetShape();
    if (shape.size() == 0 || shape[0] == 0) {
        utility::LogError("[Hashmap]: Invalid key tensor shape");
    }
    if (input_keys.GetDevice() != GetDevice()) {
        utility::LogError(
                "[Hashmap]: Incompatible device, expected {}, but got {}",
                GetDevice().ToString(), input_keys.GetDevice().ToString());
    }

    int64_t count = shape[0];
    output_masks = Tensor({count}, Dtype::Bool, GetDevice());

    return Erase(input_keys.GetDataPtr(),
                 static_cast<bool*>(output_masks.GetDataPtr()), count);
}

size_t Hashmap::GetIterators(iterator_t* output_iterators) {
    return device_hashmap_->GetIterators(output_iterators);
}

void Hashmap::UnpackIterators(const iterator_t* input_iterators,
                              const bool* input_masks,
                              void* output_keys,
                              void* output_values,
                              size_t count) {
    return device_hashmap_->UnpackIterators(input_iterators, input_masks,
                                            output_keys, output_values, count);
}

void Hashmap::AssignIterators(iterator_t* input_iterators,
                              const bool* input_masks,
                              const void* input_values,
                              size_t count) {
    return device_hashmap_->AssignIterators(input_iterators, input_masks,
                                            input_values, count);
}

size_t Hashmap::Size() const { return device_hashmap_->Size(); }

std::vector<size_t> Hashmap::BucketSizes() const {
    return device_hashmap_->BucketSizes();
}

float Hashmap::LoadFactor() const { return device_hashmap_->LoadFactor(); }

void Hashmap::AssertKeyDtype(const Dtype& dtype_key) const {
    if (dtype_key != dtype_key_) {
        utility::LogError(
                "[Hashmap] Inconsistent key dtype, expected {}, but got {}",
                dtype_key_.ToString(), dtype_key.ToString());
    }
}

void Hashmap::AssertValueDtype(const Dtype& dtype_val) const {
    if (dtype_val != dtype_val_) {
        utility::LogError(
                "[Hashmap] Inconsistent value dtype, expected {}, but got "
                "{}",
                dtype_val_.ToString(), dtype_val.ToString());
    }
}
}  // namespace core
}  // namespace open3d
