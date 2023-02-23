// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
