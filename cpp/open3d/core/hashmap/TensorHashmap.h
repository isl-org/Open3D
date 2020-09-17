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
class TensorHashmap {
public:
    /// Don't specify anything before real insertion, since many users only
    /// insert once.
    TensorHashmap(Dtype key_dtype,
                  Dtype val_dtype,
                  int64_t key_dim,
                  int64_t val_dim,
                  const Device& device = Device("CPU:0"))
        : hashmap_(nullptr),
          key_dtype_(key_dtype),
          val_dtype_(val_dtype),
          key_dim_(key_dim),
          val_dim_(val_dim),
          device_(device){};

    TensorHashmap(const Tensor& coords,
                  const Tensor& values,
                  bool insert = true);

    std::pair<Tensor, Tensor> Find(const Tensor& coords);
    std::pair<Tensor, Tensor> Insert(const Tensor& coords,
                                     const Tensor& values);
    Tensor Assign(const Tensor& coords, const Tensor& values);

    static std::pair<Tensor, Tensor> Unique(const Tensor& tensor);

protected:
    std::shared_ptr<Hashmap> hashmap_;

    Dtype key_dtype_;
    Dtype val_dtype_;

    int64_t key_dim_;
    int64_t val_dim_;

    Device device_;
};

}  // namespace core
}  // namespace open3d
