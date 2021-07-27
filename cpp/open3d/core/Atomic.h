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

#pragma once

#include <cstdint>

#ifdef MSC_VER
#include <intrin.h>
#pragma intrinsic(_InterlockedExchangeAdd)
#pragma intrinsic(_InterlockedExchangeAdd64)
#endif

namespace open3d {
namespace core {

/// Adds \p val to the value stored at \p address and returns the previous
/// stored value as an atomic operation. This function does not impose any
/// ordering on concurrent memory accesses.
/// \warning This function will treat all values as signed integers on Windows!
inline uint32_t AtomicFetchAddRelaxed(uint32_t* address, uint32_t val) {
#ifdef __GNUC__
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#elif _MSC_VER
    static_assert(sizeof(long) == sizeof(uint32_t),
                  "Expected long to be a 32 bit type");
    return static_cast<uint32_t>(_InterlockedExchangeAdd(
            reinterpret_cast<long*>(address), static_cast<long>(val)));
#else
    static_assert(false, "AtomicFetchAddRelaxed not implemented for platform");
#endif
}

/// Adds \p val to the value stored at \p address and returns the previous
/// stored value as an atomic operation. This function does not impose any
/// ordering on concurrent memory accesses.
/// \warning This function will treat all values as signed integers on Windows!
inline uint64_t AtomicFetchAddRelaxed(uint64_t* address, uint64_t val) {
#ifdef __GNUC__
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#elif _MSC_VER
    return static_cast<uint64_t>(_InterlockedExchangeAdd64(
            reinterpret_cast<int64_t*>(address), static_cast<int64_t>(val)));
#else
    static_assert(false, "AtomicFetchAddRelaxed not implemented for platform");
#endif
}

}  // namespace core
}  // namespace open3d
