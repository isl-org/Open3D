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

typedef uint32_t addr_t;

struct iterator_t {
    OPEN3D_HOST_DEVICE iterator_t() : first(nullptr), second(nullptr) {}
    OPEN3D_HOST_DEVICE iterator_t(void* key_ptr, void* value_ptr)
        : first(key_ptr), second(value_ptr) {}

    void* first;
    void* second;
};

template <typename First, typename Second>
struct Pair {
    First first;
    Second second;
    OPEN3D_HOST_DEVICE Pair() {}
    OPEN3D_HOST_DEVICE Pair(const First& _first, const Second& _second)
        : first(_first), second(_second) {}
};

template <typename First, typename Second>
OPEN3D_HOST_DEVICE Pair<First, Second> make_pair(const First& _first,
                                                 const Second& _second) {
    return Pair<First, Second>(_first, _second);
}

}  // namespace core
}  // namespace open3d
