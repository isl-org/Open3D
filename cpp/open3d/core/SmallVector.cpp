// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
//
// Adapted for Open3D.
// Commit 75e164f61d391979b4829bf2746a5d74b94e95f2 2022-01-21
// Documentation:
// https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h
//
//===- llvm/ADT/SmallVector.cpp - 'Normally small' vectors ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallVector class.
//
//===----------------------------------------------------------------------===//

#include "open3d/core/SmallVector.h"

#define LLVM_ENABLE_EXCEPTIONS

#ifdef LLVM_ENABLE_EXCEPTIONS
#include <stdexcept>
#endif
#include <string>
namespace open3d {
namespace core {

// Check that no bytes are wasted and everything is well-aligned.
namespace {
// These structures may cause binary compat warnings on AIX. Suppress the
// warning since we are only using these types for the static assertions below.
#if defined(_AIX)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waix-compat"
#endif
struct Struct16B {
    alignas(16) void *X;
};
struct Struct32B {
    alignas(32) void *X;
};
#if defined(_AIX)
#pragma GCC diagnostic pop
#endif
}  // namespace
static_assert(sizeof(SmallVector<void *, 0>) ==
                      sizeof(unsigned) * 2 + sizeof(void *),
              "wasted space in SmallVector size 0");
static_assert(alignof(SmallVector<Struct16B, 0>) >= alignof(Struct16B),
              "wrong alignment for 16-byte aligned T");
static_assert(alignof(SmallVector<Struct32B, 0>) >= alignof(Struct32B),
              "wrong alignment for 32-byte aligned T");
static_assert(sizeof(SmallVector<Struct16B, 0>) >= alignof(Struct16B),
              "missing padding for 16-byte aligned T");
static_assert(sizeof(SmallVector<Struct32B, 0>) >= alignof(Struct32B),
              "missing padding for 32-byte aligned T");
static_assert(sizeof(SmallVector<void *, 1>) ==
                      sizeof(unsigned) * 2 + sizeof(void *) * 2,
              "wasted space in SmallVector size 1");

static_assert(sizeof(SmallVector<char, 0>) ==
                      sizeof(void *) * 2 + sizeof(void *),
              "1 byte elements have word-sized type for size and capacity");

/// Report that MinSize doesn't fit into this vector's size type. Throws
/// std::length_error or calls report_fatal_error.
[[noreturn]] static void report_size_overflow(size_t MinSize, size_t MaxSize);
static void report_size_overflow(size_t MinSize, size_t MaxSize) {
    std::string Reason = "SmallVector unable to grow. Requested capacity (" +
                         std::to_string(MinSize) +
                         ") is larger than maximum value for size type (" +
                         std::to_string(MaxSize) + ")";
#ifdef LLVM_ENABLE_EXCEPTIONS
    throw std::length_error(Reason);
#else
    report_fatal_error(Twine(Reason));
#endif
}

/// Report that this vector is already at maximum capacity. Throws
/// std::length_error or calls report_fatal_error.
[[noreturn]] static void report_at_maximum_capacity(size_t MaxSize);
static void report_at_maximum_capacity(size_t MaxSize) {
    std::string Reason =
            "SmallVector capacity unable to grow. Already at maximum size " +
            std::to_string(MaxSize);
#ifdef LLVM_ENABLE_EXCEPTIONS
    throw std::length_error(Reason);
#else
    report_fatal_error(Twine(Reason));
#endif
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
static size_t getNewCapacity(size_t MinSize, size_t TSize, size_t OldCapacity) {
    constexpr size_t MaxSize = std::numeric_limits<Size_T>::max();

    // Ensure we can fit the new capacity.
    // This is only going to be applicable when the capacity is 32 bit.
    if (MinSize > MaxSize) report_size_overflow(MinSize, MaxSize);

    // Ensure we can meet the guarantee of space for at least one more element.
    // The above check alone will not catch the case where grow is called with a
    // default MinSize of 0, but the current capacity cannot be increased.
    // This is only going to be applicable when the capacity is 32 bit.
    if (OldCapacity == MaxSize) report_at_maximum_capacity(MaxSize);

    // In theory 2*capacity can overflow if the capacity is 64 bit, but the
    // original capacity would never be large enough for this to be a problem.
    size_t NewCapacity = 2 * OldCapacity + 1;  // Always grow.
    return std::min(std::max(NewCapacity, MinSize), MaxSize);
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
void *SmallVectorBase<Size_T>::mallocForGrow(size_t MinSize,
                                             size_t TSize,
                                             size_t &NewCapacity) {
    NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
    return open3d::core::safe_malloc(NewCapacity * TSize);
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
void SmallVectorBase<Size_T>::grow_pod(void *FirstEl,
                                       size_t MinSize,
                                       size_t TSize) {
    size_t NewCapacity =
            getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
    void *NewElts;
    if (BeginX == FirstEl) {
        NewElts = safe_malloc(NewCapacity * TSize);

        // Copy the elements over.  No need to run dtors on PODs.
        memcpy(NewElts, this->BeginX, size() * TSize);
    } else {
        // If this wasn't grown from the inline copy, grow the allocated space.
        NewElts = safe_realloc(this->BeginX, NewCapacity * TSize);
    }

    this->BeginX = NewElts;
    this->Capacity = NewCapacity;
}

template class SmallVectorBase<uint32_t>;

// Disable the uint64_t instantiation for 32-bit builds.
// Both uint32_t and uint64_t instantiations are needed for 64-bit builds.
// This instantiation will never be used in 32-bit builds, and will cause
// warnings when sizeof(Size_T) > sizeof(size_t).
#if SIZE_MAX > UINT32_MAX
template class SmallVectorBase<uint64_t>;

// Assertions to ensure this #if stays in sync with SmallVectorSizeType.
static_assert(sizeof(SmallVectorSizeType<char>) == sizeof(uint64_t),
              "Expected SmallVectorBase<uint64_t> variant to be in use.");
#else
static_assert(sizeof(SmallVectorSizeType<char>) == sizeof(uint32_t),
              "Expected SmallVectorBase<uint32_t> variant to be in use.");
#endif
}  // namespace core
}  // namespace open3d
