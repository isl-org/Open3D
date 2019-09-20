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

#include "Open3D/Container/Device.h"
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/Shape.h"

// TODO: move the contents of this folder to "Open3D/src"?
//       currently they are in "open3d" top namespace but under "TensorArray"
//       folder
namespace open3d {

template <typename T>
class Blob {
public:
    Blob(const Shape& shape, const Device& device)
        : shape_(shape),
          device_(device),
          num_elements_(shape.NumElements()),
          byte_size_(sizeof(T) * shape.NumElements()) {
        v_ = static_cast<T*>(
                MemoryManager::Allocate(byte_size_, device_.DeviceTypeStr()));
    }

    ~Blob() { MemoryManager::Free(v_); };

public:
    T* v_ = nullptr;

public:
    Shape shape_ = {};
    Device device_;
    size_t num_elements_ = 0;  // Num elements
    size_t byte_size_ = 0;     // Total num bytes
};

}  // namespace open3d
