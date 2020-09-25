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

/// Built-in flags
static constexpr uint32_t BASE_UNIT_SIZE = 32;
static constexpr uint32_t EMPTY_SLAB_PTR = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_PAIR_PTR = 0xFFFFFFFF;
static constexpr uint32_t HEAD_SLAB_PTR = 0xFFFFFFFE;

/// Queries
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;

/// Warp operations
//
// REVIEW:
// Let's add comments / rename / remove some of these value-32 variables.
// Currently they could be a bit confusing. The following comments are based on
// my current understanding of the code, so please correct me if it is not
// accurate.
//
// What the `SIZE` refers to can be confusing. It's not clear it means number of
// elements, number of 32-bit integers, number of bytes or something else. It
// would be nice to be more explicit.
// - MEM_UNIT_SIZE_ == 32 means that a memory unit (a slab) has the size
//   of "32" 32-bit integers (total 128 Bytes). This "size" refers to the
//   size measured in the number of 32-bit integers.
//
// My suggestion is to standardize to the convention. Here are some example
// ideas:
//     - Instead of SIZE, always use BYTE_SIZE. Avoid measuring memory size with
//       number of 32-bit integers.
//     - SIZE or BYTE_SIZE always refer to byte size, i.e. size in bytes. So we
//       don't measure things in number of 32-bit integers.
//     - NUM always refer to "count". E.g. we can rename "WARP_SIZE" to
//       "NUM_THREADS_PER_WARP".
//
// [List of variables with value 32]
// - MAX_KEY_BYTESIZE         : OK.
// - BASE_UNIT_SIZE           : ?
// - WARP_WIDTH               : Duplicate of WARP_SIZE?
// - NUM_SUPER_BLOCKS_        : OK.
// - NUM_BITMAP_PER_MEM_BLOCK_: This is determined by the fact that we hard-code
//                              NUM_MEM_UNITS_PER_BLOCK_ and we use 32-bit bit
//                              maps. So this is NUM_MEM_UNITS_PER_BLOCK_ / 32.
// - BITMAP_SIZE_             : It's a bit convoluted, but it should be always
//                              the same as NUM_BITMAP_PER_MEM_BLOCK_.
//                              - BITMAP_SIZE_ is actually "total bitmap size
//                                per block", measured in number of intergers.
//                              - Since each slab needs 1 bit, "total bitmap
//                                size per block" is NUM_MEM_UNITS_PER_BLOCK_
//                                bits.
//                              - Since we're using 32-bit integers to store the
//                                bit map, we need NUM_MEM_UNITS_PER_BLOCK_ / 32
//                                integers. This means BITMAP_SIZE_ is always
//                                numerically the same as
//                                NUM_BITMAP_PER_MEM_BLOCK_. This equality
//                                doesn't change even if chage the parameters,
//                                e.g. using 4096 mem units per block,
//                                NUM_BITMAP_PER_MEM_BLOCK_ == BITMAP_SIZE_
//                                == 64.
// - WARP_SIZE                : OK. Determined by the device.
// - MEM_UNIT_SIZE_           : This is actually sizeof(Slab) measured in number
//                              of 32-bit integers. It would be nice to somehow
//                              unify this with the Slab struct.

static constexpr uint32_t WARP_WIDTH = 32;
static constexpr uint32_t BLOCKSIZE_ = 128;

/// bits:   31 | 30 | ... | 3 | 2 | 1 | 0
static constexpr uint32_t ACTIVE_LANES_MASK = 0xFFFFFFFF;
static constexpr uint32_t PAIR_PTR_LANES_MASK = 0x7FFFFFFF;
static constexpr uint32_t NEXT_SLAB_PTR_LANE = 31;

static constexpr uint32_t NULL_ITERATOR = 0xFFFFFFFF;

static constexpr uint32_t NUM_SUPER_BLOCKS_ = 32;

static constexpr uint32_t LOG_NUM_MEM_BLOCKS_ = 2;
static constexpr uint32_t NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ = 4;
static constexpr uint32_t NUM_MEM_UNITS_PER_BLOCK_ = 1024;

// REVIEW: NUM_BITMAP_PER_MEM_BLOCK_ not used
static constexpr uint32_t NUM_BITMAP_PER_MEM_BLOCK_ = 32;
static constexpr uint32_t BITMAP_SIZE_ = 32;
static constexpr uint32_t WARP_SIZE = 32;
static constexpr uint32_t MEM_UNIT_SIZE_ = 32;
static constexpr uint32_t SUPER_BLOCK_BIT_OFFSET_ALLOC_ = 27;
static constexpr uint32_t MEM_BLOCK_BIT_OFFSET_ALLOC_ = 10;
static constexpr uint32_t MEM_UNIT_BIT_OFFSET_ALLOC_ = 5;

static constexpr uint32_t MEM_BLOCK_SIZE_ =
        NUM_MEM_UNITS_PER_BLOCK_ * MEM_UNIT_SIZE_;
static constexpr uint32_t SUPER_BLOCK_SIZE_ =
        ((BITMAP_SIZE_ + MEM_BLOCK_SIZE_) * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
static constexpr uint32_t MEM_BLOCK_OFFSET_ =
        (BITMAP_SIZE_ * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);

}  // namespace core
}  // namespace open3d
