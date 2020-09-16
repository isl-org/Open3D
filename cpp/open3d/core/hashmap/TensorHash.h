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
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Hashmap.h"

namespace open3d {
namespace core {
class TensorHash {
public:
    /// Don't specify anything before real insertion, since many users only
    /// insert once.
    TensorHash(Dtype key_type,
               Dtype value_type,
               int64_t key_dim,
               int64_t value_dim,
               Device device = Device("CPU:0"))
        : hashmap_(nullptr),
          key_type_(key_type),
          value_type_(value_type),
          key_dim_(key_dim),
          value_dim_(value_dim),
          device_(device){};

    TensorHash(Tensor coords, Tensor values, bool insert = true);

    /// <Value, Mask>
    std::pair<Tensor, Tensor> Query(Tensor coords);
    /// <Key, Mask>
    std::pair<Tensor, Tensor> Insert(Tensor coords, Tensor values);
    /// Mask
    Tensor Assign(Tensor coords, Tensor values);

    static std::pair<Tensor, Tensor> Unique(const Tensor &tensor);

protected:
    std::shared_ptr<Hashmap> hashmap_;

    Dtype key_type_;
    Dtype value_type_;

    int64_t key_dim_;
    int64_t value_dim_;

    Device device_;
};

}  // namespace core
}  // namespace open3d
