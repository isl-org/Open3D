// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

// These classes implement functors that can be passed to the neighbor search
// functions.

template <class T, class TIndex>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(torch::DeviceType device_type, int device_idx)
        : device_type(device_type), device_idx(device_idx) {}

    void AllocIndices(TIndex** ptr, size_t num) {
        neighbors_index = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<TIndex>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_index.data_ptr<TIndex>();
    }

    void AllocDistances(T** ptr, size_t num) {
        neighbors_distance = torch::empty(
                {int64_t(num)}, torch::dtype(ToTorchDtype<T>())
                                        .device(device_type, device_idx));
        *ptr = neighbors_distance.data_ptr<T>();
    }

    const TIndex* IndicesPtr() const {
        return neighbors_index.data_ptr<TIndex>();
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
