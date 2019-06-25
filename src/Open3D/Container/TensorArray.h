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
#include <string>

#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/Shape.h"

namespace open3d {

template <typename T>
class TensorArray {
public:
    TensorArray(const Shape& tensor_shape,
                size_t max_size,
                const std::string& device = "cpu")
        : tensor_shape_(tensor_shape),
          max_size_(max_size),
          device_(device),
          curr_size_(0) {
        if (device == "cpu" || device == "gpu") {
            v_ = static_cast<T*>(MemoryManager::Allocate(
                    TensorByteSize() * max_size_, device_));
        } else {
            throw std::runtime_error("Unrecognized device");
        }
    }

    ~TensorArray() { MemoryManager::Free(v_); };

    size_t TensorByteSize() const {
        return sizeof(T) * tensor_shape_.NumElements();
    }

public:
    T* v_;
    size_t curr_size_;
    size_t max_size_;

public:
    const Shape tensor_shape_;
    const std::string device_;
};

}  // namespace open3d
