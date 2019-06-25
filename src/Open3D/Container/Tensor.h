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

#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/Shape.h"

// TODO: move the contents of this folder to "Open3D/src"?
//       currently they are in "open3d" top namespace but under "TensorArray"
//       folder
namespace open3d {

template <typename T>
class Tensor {
public:
    Tensor(const Shape& shape, const std::string& device = "cpu")
        : shape_(shape),
          device_(device),
          num_elements_(shape.NumElements()),
          byte_size_(sizeof(T) * shape.NumElements()) {
        if (device == "cpu" || device == "gpu") {
            v_ = static_cast<T*>(MemoryManager::Allocate(byte_size_, device_));
        } else {
            throw std::runtime_error("Unrecognized device");
        }
    }

    Tensor(const std::vector<T>& init_vals,
           const Shape& shape,
           const std::string& device = "cpu")
        : Tensor(shape, device) {
        if (init_vals.size() != num_elements_) {
            throw std::runtime_error(
                    "Tensor initialization values' size does not match the "
                    "shape.");
        }

        if (device == "cpu" || device == "gpu") {
            MemoryManager::CopyTo(v_, init_vals.data(), byte_size_);
        } else if (device == "gpu") {
            throw std::runtime_error("Unimplemented");
        } else {
            throw std::runtime_error("Unrecognized device");
        }
    }

    ~Tensor() { MemoryManager::Free(v_); };

    std::vector<T> ToStdVector() const {
        std::vector<T> vec(num_elements_);
        MemoryManager::CopyTo(vec.data(), v_, byte_size_);
        return vec;
    }

public:
    T* v_;

public:
    const Shape shape_;
    const std::string device_;
    const size_t num_elements_;  // Num elements
    const size_t byte_size_;     // Num bytes
};

}  // namespace open3d
