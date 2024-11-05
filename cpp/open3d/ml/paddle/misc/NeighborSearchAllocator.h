// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/paddle/PaddleHelper.h"

// These classes implement functors that can be passed to the neighbor search
// functions.

template <class T, class TIndex>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(paddle::Place place) : place(place) {}

    void AllocIndices(TIndex** ptr, size_t num) {
        if (num == 0) {
            neighbors_index = InitializedEmptyTensor<TIndex>({0}, place);
        } else {
            neighbors_index = paddle::empty(
                    {int64_t(num)}, paddle::DataType(ToPaddleDtype<TIndex>()),
                    place);
        }
        *ptr = neighbors_index.data<TIndex>();
    }

    void AllocDistances(T** ptr, size_t num) {
        if (num == 0) {
            neighbors_distance = InitializedEmptyTensor<T>({0}, place);
        } else {
            neighbors_distance =
                    paddle::empty({int64_t(num)},
                                  paddle::DataType(ToPaddleDtype<T>()), place);
        }
        *ptr = neighbors_distance.data<T>();
    }

    const TIndex* IndicesPtr() const { return neighbors_index.data<TIndex>(); }

    const T* DistancesPtr() const { return neighbors_distance.data<T>(); }

    const paddle::Tensor& NeighborsIndex() const { return neighbors_index; }
    const paddle::Tensor& NeighborsDistance() const {
        return neighbors_distance;
    }

private:
    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_distance;
    paddle::Place place;
};
