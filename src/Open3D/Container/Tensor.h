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
#include <memory>
#include <string>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/Shape.h"

namespace open3d {

class Tensor {
public:
    Tensor(const Shape& shape,
           const Dtype& dtype,
           const Device& device = Device("CPU:0"))
        : shape_(shape),
          dtype_(dtype),
          device_(device),
          blob_(std::make_shared<Blob>(ByteSize(), device)) {}

    size_t ByteSize() const {
        return shape_.NumElements() * DtypeUtil::ByteSize(dtype_);
    }

    std::shared_ptr<Blob> GetBlob() const { return blob_; }

    Shape GetShape() const { return shape_; }

    Dtype GetDtype() const { return dtype_; }

    Device GetDevice() const { return device_; }

    // Tensor(const std::vector<T>& init_vals,
    //        const Shape& shape,
    //        const std::string& device = "CPU")
    //     : Tensor(shape, device) {
    //     if (init_vals.size() != num_elements_) {
    //         throw std::runtime_error(
    //                 "Tensor initialization values' size does not match the "
    //                 "shape.");
    //     }

    //     if (device == "CPU" || device == "GPU") {
    //         MemoryManager::CopyTo(v_, init_vals.data(), byte_size_);
    //     } else if (device == "GPU") {
    //         throw std::runtime_error("Unimplemented");
    //     } else {
    //         throw std::runtime_error("Unrecognized device");
    //     }
    // }

    // ~Tensor() { MemoryManager::Free(v_); };

    // std::vector<T> ToStdVector() const {
    //     std::vector<T> vec(num_elements_);
    //     MemoryManager::CopyTo(vec.data(), v_, byte_size_);
    //     return vec;
    // }

protected:
    Shape shape_;
    Dtype dtype_;
    Device device_;
    std::shared_ptr<Blob> blob_;
};

}  // namespace open3d
