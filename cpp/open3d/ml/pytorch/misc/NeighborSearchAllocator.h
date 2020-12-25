// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
//

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

// These classes implement functors that can be passed to the neighbor search
// functions.

template <class T>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(torch::DeviceType device_type, int device_idx)
        : device_type(device_type), device_idx(device_idx) {}

    void AllocIndices(int32_t** ptr, size_t num) {
        neighbors_index = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<int32_t>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_index.data_ptr<int32_t>();
    }

    void AllocDistances(T** ptr, size_t num) {
        neighbors_distance = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<T>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_distance.data_ptr<T>();
    }

    const int32_t* IndicesPtr() const {
        return neighbors_index.data_ptr<int32_t>();
    }

    const T* DistancesPtr() const { return neighbors_distance.data_ptr<T>(); }

    const torch::Tensor& NeighborsIndex() const { return neighbors_index; }
    const torch::Tensor& NeighborsDistance() const {
        return neighbors_distance;
    }

private:
    torch::Tensor neighbors_index;
    torch::Tensor neighbors_distance;
    torch::DeviceType device_type;
    int device_idx;
};
