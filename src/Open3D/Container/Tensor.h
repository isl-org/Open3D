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

    template <typename T>
    Tensor(const std::vector<T>& init_vals,
           const Shape& shape,
           const Dtype& dtype,
           const Device& device = Device("CPU:0"))
        : Tensor(shape, dtype, device) {
        // Check number of elements
        if (init_vals.size() != shape_.NumElements()) {
            utility::LogFatal(
                    "Tensor initialization values' size does not match the "
                    "shape.\n");
        }

        // Check data types
        if (DtypeUtil::FromType<T>() != dtype_) {
            utility::LogFatal(
                    "Init values have type {} but Tensor has type {}\n",
                    DtypeUtil::ToString(DtypeUtil::FromType<T>()),
                    DtypeUtil::ToString(dtype_));
        }
        if (DtypeUtil::ByteSize(dtype_) != sizeof(T)) {
            utility::LogFatal(
                    "Internal error: element size mismatch {} != {}\n",
                    DtypeUtil::ByteSize(dtype_), sizeof(T));
        }

        // Copy data to blob
    }

    size_t ByteSize() const {
        return shape_.NumElements() * DtypeUtil::ByteSize(dtype_);
    }

public:
    std::shared_ptr<Blob> GetBlob() const { return blob_; }

    Shape GetShape() const { return shape_; }

    Dtype GetDtype() const { return dtype_; }

    Device GetDevice() const { return device_; }

protected:
    Shape shape_;
    Dtype dtype_;
    Device device_;
    std::shared_ptr<Blob> blob_;
};

}  // namespace open3d
