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

#include "open3d/core/Device.h"
#include "open3d/core/MemoryManager.h"

namespace open3d {
namespace core {

/// Class satisfying the Allocator requirements defined by the C++ standard.
/// This bridge makes the MemoryManager interface accessible to all classes
/// and containers that use the standard Allocator interface.
///
/// This is particularly useful to allocate (potentially cached) GPU memory
/// or different types of memory depending on the provided device.
template <typename T>
class StdAllocator {
public:
    /// T.
    using value_type = T;

    /// Default constructor.
    StdAllocator() = default;

    /// Constructor from device.
    explicit StdAllocator(const Device& device) : device_(device) {}

    /// Default copy constructor.
    StdAllocator(const StdAllocator&) = default;

    /// Default copy assignment operator.
    StdAllocator& operator=(const StdAllocator&) = default;

    /// Default move constructor.
    StdAllocator(StdAllocator&&) = default;

    /// Default move assignment operator.
    StdAllocator& operator=(StdAllocator&&) = default;

    /// Rebind copy constructor.
    template <typename U>
    StdAllocator(const StdAllocator<U>& other) : device_(other.device_) {}

    /// Allocates memory of size \p n.
    T* allocate(std::size_t n) {
        return static_cast<T*>(MemoryManager::Malloc(n * sizeof(T), device_));
    }

    /// Deallocates memory from pointer \p p of size \p n .
    void deallocate(T* p, std::size_t n) { MemoryManager::Free(p, device_); }

    /// Returns true if the instances are equal, false otherwise.
    bool operator==(const StdAllocator& other) const {
        return device_ == other.device_;
    }

    /// Returns true if the instances are not equal, false otherwise.
    bool operator!=(const StdAllocator& other) const {
        return !operator==(other);
    }

    /// Returns the device on which memory is allocated.
    Device GetDevice() const { return device_; }

private:
    // Allow access in rebind constructor.
    template <typename T2>
    friend class StdAllocator;

    Device device_;
};

}  // namespace core
}  // namespace open3d
