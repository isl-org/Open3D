// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    explicit StdAllocator(int device_id) : device_id_(device_id) {}

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
    StdAllocator(const StdAllocator<U>& other) : device_id_(other.device_id_) {}

    /// Allocates memory of size \p n.
    T* allocate(std::size_t n) {
        void* ptr = MemoryManager::Malloc(
                n * sizeof(T), Device(Device::DeviceType::CUDA, device_id_));
        return static_cast<T*>(ptr);
    }

    /// Deallocates memory from pointer \p p of size \p n .
    void deallocate(T* p, std::size_t n) {
        MemoryManager::Free(p, Device(Device::DeviceType::CUDA, device_id_));
    }

    /// Returns true if the instances are equal, false otherwise.
    bool operator==(const StdAllocator& other) const {
        return device_id_ == other.device_id_;
    }

    /// Returns true if the instances are not equal, false otherwise.
    bool operator!=(const StdAllocator& other) const {
        return !operator==(other);
    }

    /// Returns the device on which memory is allocated.
    int GetDeviceID() const { return device_id_; }

private:
    // Allow access in rebind constructor.
    template <typename T2>
    friend class StdAllocator;

    int device_id_;
};

}  // namespace core
}  // namespace open3d
