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
//
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
#include <cstdint>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

static constexpr uint32_t MAX_KEY_BYTESIZE = 32;

typedef uint32_t ptr_t;
struct iterator_t {
    OPEN3D_HOST_DEVICE iterator_t() : first(nullptr), second(nullptr) {}
    OPEN3D_HOST_DEVICE iterator_t(void* key_ptr, void* value_ptr)
        : first(key_ptr), second(value_ptr) {}

    void* first;
    void* second;
};

typedef uint64_t (*hash_t)(uint8_t*, uint32_t);

/// Internal Hashtable Node: (31 units and 1 next ptr) representation.
/// \member kv_pair_ptrs:
/// Each element is an internal ptr to a kv pair managed by the
/// InternalMemoryManager. Can be converted to a real ptr.
/// \member next_slab_ptr:
/// An internal ptr managed by InternalNodeManager.
class Slab {
public:
    ptr_t kv_pair_ptrs[31];
    ptr_t next_slab_ptr;
};

template <typename Key, typename Value>
struct Pair {
    Key first;
    Value second;
    OPEN3D_HOST_DEVICE Pair() {}
    OPEN3D_HOST_DEVICE Pair(const Key& key, const Value& value)
        : first(key), second(value) {}
};

template <typename Key, typename Value>
OPEN3D_HOST_DEVICE Pair<Key, Value> make_pair(const Key& key,
                                              const Value& value) {
    return Pair<Key, Value>(key, value);
}

}  // namespace core
}  // namespace open3d
