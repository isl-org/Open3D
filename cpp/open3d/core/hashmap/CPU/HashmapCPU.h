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

#pragma once

// Interface for the CPU hashmap. Separated from HashmapCPU.hpp for brevity.

#include <unordered_map>
#include "open3d/core/hashmap/HashmapBase.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CPUHashmap : public Hashmap<Hash, KeyEq> {
public:
    ~CPUHashmap();

    CPUHashmap(size_t init_buckets,
               size_t init_capacity,
               size_t dsize_key,
               size_t dsize_value,
               Device device);

    void Rehash(size_t buckets);

    void Insert(const void* input_keys,
                const void* input_values,
                iterator_t* output_iterators,
                bool* output_masks,
                size_t count);

    void Find(const void* input_keys,
              iterator_t* output_iterators,
              bool* output_masks,
              size_t count);

    void Erase(const void* input_keys, bool* output_masks, size_t count);

    size_t GetIterators(iterator_t* output_iterators);

    void UnpackIterators(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         size_t count);

    void AssignIterators(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         size_t count);

    std::vector<size_t> BucketSizes();
    float LoadFactor();

private:
    std::shared_ptr<std::unordered_map<void*, void*, Hash, KeyEq>> impl_;

    // Valid kv_pairs
    std::vector<iterator_t> kv_pairs_;
};
}  // namespace core
}  // namespace open3d
