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

#pragma once

#include <cstddef>
#include <iostream>
#include <string>

#include "Open3D/Core/Device.h"
#include "Open3D/Core/MemoryManager.h"

namespace open3d {

/// Blob class refers to a blob of memory in device or host.
///
/// Usually a Blob is constructed by specifying the blob size and device, memory
/// allocation happens during the Blob's construction.
///
/// A Blob's buffer can also be managed by an external memory manager. In this
/// case, a deleter function is needed to notify the external memory manager
/// that the memory is no longer needed. It does not make sense to infer the
/// total buffer size. For example, if a Tensor has a negative stride size, it
/// is necessary to access memory addresses smaller than Blob's beginning memory
/// address. The only responsibility for Blob is to hold the beginning
/// memory address and it's up to the user to access any addresses around it.
///
/// In summary:
/// - A Blob does not know about its memory size after construction.
/// - A Blob cannot be deep-copied. However, the Tensor which owns the blob can
/// be copied.
class Blob {
public:
    /// Construct Blob on a specified device.
    ///
    /// \param byte_size Size of the blob in bytes.
    /// \param device Device where the blob resides.
    Blob(int64_t byte_size, const Device& device)
        : deleter_(nullptr),
          data_ptr_(MemoryManager::Malloc(byte_size, device)),
          device_(device) {}

    /// Construct Blob with externally managed memory.
    ///
    /// \param device Device where the blob resides.
    /// \param data_ptr Pointer the blob's beginning.
    /// \param deleter The deleter function is called at Blob's destruction to
    /// notify the external memory manager that the memory is no longer needed.
    /// It's up to the external manager to free the memory.
    Blob(const Device& device,
         void* data_ptr,
         const std::function<void(void*)>& deleter)
        : deleter_(deleter), data_ptr_(data_ptr), device_(device) {}

    ~Blob() {
        if (deleter_) {
            // Our custom deleter's void* argument is not used. The deleter
            // function itself shall handle destruction without the argument.
            // The void(void*) signature is kept to be consistent with DLPack's
            // deleter.
            deleter_(nullptr);
        } else {
            MemoryManager::Free(data_ptr_, device_);
        }
    };

    Device GetDevice() const { return device_; }

    void* GetDataPtr() { return data_ptr_; }

    const void* GetDataPtr() const { return data_ptr_; }

protected:
    /// For externally managed memory, deleter != nullptr.
    std::function<void(void*)> deleter_ = nullptr;

    /// Device data pointer.
    void* data_ptr_ = nullptr;

    /// Device context for the blob.
    Device device_;
};

}  // namespace open3d
