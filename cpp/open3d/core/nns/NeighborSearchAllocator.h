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
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {
namespace nns {

template <class T, class TIndex = int32_t>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(Device device) : device_(device) {}

    void AllocIndices(TIndex** ptr, size_t num) {
        indices_ = Tensor::Empty({int64_t(num)}, Dtype::FromType<TIndex>(),
                                 device_);
        *ptr = indices_.GetDataPtr<TIndex>();
    }

    void AllocIndices(TIndex** ptr, size_t num, TIndex value) {
        indices_ = Tensor::Full({int64_t(num)}, value,
                                Dtype::FromType<TIndex>(), device_);
        *ptr = indices_.GetDataPtr<TIndex>();
    }

    void AllocDistances(T** ptr, size_t num) {
        distances_ =
                Tensor::Empty({int64_t(num)}, Dtype::FromType<T>(), device_);
        *ptr = distances_.GetDataPtr<T>();
    }

    void AllocDistances(T** ptr, size_t num, T value) {
        distances_ = Tensor::Full({int64_t(num)}, value, Dtype::FromType<T>(),
                                  device_);
        *ptr = distances_.GetDataPtr<T>();
    }

    void AllocCounts(TIndex** ptr, size_t num) {
        counts_ = Tensor::Empty({int64_t(num)}, Dtype::FromType<TIndex>(),
                                device_);
        *ptr = counts_.GetDataPtr<TIndex>();
    }

    void AllocCounts(TIndex** ptr, size_t num, TIndex value) {
        counts_ = Tensor::Full({int64_t(num)}, value, Dtype::FromType<TIndex>(),
                               device_);
        *ptr = counts_.GetDataPtr<TIndex>();
    }

    const TIndex* IndicesPtr() const { return indices_.GetDataPtr<TIndex>(); }

    const T* DistancesPtr() const { return distances_.GetDataPtr<T>(); }

    const TIndex* CountsPtr() const { return counts_.GetDataPtr<TIndex>(); }

    const Tensor& NeighborsIndex() const { return indices_; }
    Tensor& NeighborsIndex_() { return indices_; }
    const Tensor& NeighborsDistance() const { return distances_; }
    Tensor& NeighborsDistance_() { return distances_; }
    const Tensor& NeighborsCount() const { return counts_; }

private:
    Tensor indices_;
    Tensor distances_;
    Tensor counts_;
    Device device_;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
