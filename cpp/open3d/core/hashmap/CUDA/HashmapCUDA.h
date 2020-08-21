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

// Copyright 2019 Saman Ashkiani
// Rewritten by Wei Dong 2019 - 2020
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing permissions
// and limitations under the License.

#pragma once

// Interface for the CUDA hashmap. Separated from HashmapCUDA.cuh for brevity.

#include <cassert>
#include <memory>

#include <thrust/pair.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"

#include "open3d/core/hashmap/CUDA/HashmapCUDAImpl.h"
#include "open3d/core/hashmap/HashmapBase.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CUDAHashmap : public Hashmap<Hash, KeyEq> {
public:
    ~CUDAHashmap();

    CUDAHashmap(size_t init_buckets,
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

protected:
    /// struct directly passed to kernels, cannot be a pointer
    CUDAHashmapImplContext<Hash, KeyEq> gpu_context_;

    std::shared_ptr<InternalKvPairManager> mem_mgr_;
    std::shared_ptr<InternalNodeManager> node_mgr_;
};
}  // namespace core
}  // namespace open3d
