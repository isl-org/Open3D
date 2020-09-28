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

// Device-specific, safe for current NVIDIA architectures.
static constexpr uint32_t kWarpSize = 32;

//////////////////////
// Tunable variables
//////////////////////
// Hashmap
static constexpr uint32_t kSuperBlocks = 32;
static constexpr uint32_t kBlocksPerSuperBlock = 4;
static constexpr uint32_t kBlocksPerSuperBlockInBits = 2;
static constexpr uint32_t kSlabsPerBlock = 1024;

// Misc
static constexpr uint32_t kMaxKeyByteSize = 32;
static constexpr uint32_t kThreadsPerBlock = 128;

//////////////////////
// Combination of tunable variables
//////////////////////
static constexpr uint32_t kUIntsPerBlock = kSlabsPerBlock * kWarpSize;
// We need one bitmap (32-bit) per slab (32 nodes). Number of bitmaps is equal
// to number of slabs.
static constexpr uint32_t kBitmapsPerSuperBlock =
        kBlocksPerSuperBlock * kSlabsPerBlock;
static constexpr uint32_t kUIntsPerSuperBlock =
        kBlocksPerSuperBlock * kUIntsPerBlock + kBitmapsPerSuperBlock;

//////////////////////
// Non-tunable variables
//////////////////////
// Mask offsets
static constexpr uint32_t kSuperBlockMaskBits = 27;
static constexpr uint32_t kBlockMaskBits = 10;
static constexpr uint32_t kSlabMaskBits = 5;

// Masks & flags
static constexpr uint32_t kSyncLanesMask = 0xFFFFFFFF;
static constexpr uint32_t kNodePtrLanesMask = 0x7FFFFFFF;
static constexpr uint32_t kNextSlabPtrLaneId = 31;

static constexpr uint32_t kHeadSlabAddr = 0xFFFFFFFE;

static constexpr uint32_t kEmptySlabAddr = 0xFFFFFFFF;
static constexpr uint32_t kEmptyNodeAddr = 0xFFFFFFFF;
static constexpr uint32_t kNullAddr = 0xFFFFFFFF;

static constexpr uint32_t kNotFoundFlag = 0xFFFFFFFF;

#define MEMCPY_AS_INTS(dst, src, num_bytes)              \
    auto dst_in_int = reinterpret_cast<int*>(dst);       \
    auto src_in_int = reinterpret_cast<const int*>(src); \
    int count_in_int = num_bytes / sizeof(int);          \
    for (int i = 0; i < count_in_int; ++i) {             \
        dst_in_int[i] = src_in_int[i];                   \
    }

}  // namespace core
}  // namespace open3d
