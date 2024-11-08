// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/impl/misc/InvertNeighborsList.cuh"
#include "open3d/ml/paddle/PaddleHelper.h"

template <class TIndex, class TAttr>
std::vector<paddle::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        const paddle::Tensor& inp_neighbors_index,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& inp_neighbors_attributes) {
    auto place = inp_neighbors_index.place();
    paddle::Tensor neighbors_index =
            paddle::empty(inp_neighbors_index.shape(),
                          paddle::DataType(ToPaddleDtype<TIndex>()), place);
    paddle::Tensor neighbors_row_splits = paddle::empty(
            {num_points + 1}, paddle::DataType(paddle::DataType::INT64), place);
    paddle::Tensor neighbors_attributes =
            paddle::empty_like(inp_neighbors_attributes);

    // maybe this can use torch's impl way ?
    auto stream = inp_neighbors_index.stream();
    // -1 means current global place
    auto cuda_place_props = phi::backends::gpu::GetDeviceProperties(-1);
    const int texture_alignment = cuda_place_props.textureAlignment;

    int num_attributes;
    if (inp_neighbors_attributes.shape()[0] == 0) {
        std::cout << inp_neighbors_attributes.dtype() << std::endl;
        num_attributes = 0;
        neighbors_attributes =
                InitializedEmptyTensor(inp_neighbors_attributes.dtype(),
                                       inp_neighbors_attributes.shape(),
                                       inp_neighbors_attributes.place());
    } else {
        num_attributes = 1;
        for (int i = 1; i < inp_neighbors_attributes.dims().size(); ++i)
            num_attributes *= inp_neighbors_attributes.shape()[i];
    }

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data<TAttr>() : nullptr,
            num_attributes, inp_neighbors_row_splits.data<int64_t>(),
            inp_neighbors_row_splits.shape()[0] - 1,
            neighbors_index.data<TIndex>(),
            num_attributes ? neighbors_attributes.data<TAttr>() : nullptr,
            neighbors_index.shape()[0],
            neighbors_row_splits.data<int64_t>(),  // NOLINT
            neighbors_row_splits.shape()[0] - 1);

    auto temp_tensor = CreateTempTensor(temp_size, place, &temp_ptr);

    // actually invert the list
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data<TAttr>() : nullptr,
            num_attributes, inp_neighbors_row_splits.data<int64_t>(),
            inp_neighbors_row_splits.shape()[0] - 1,
            neighbors_index.data<TIndex>(),
            num_attributes ? neighbors_attributes.data<TAttr>() : nullptr,
            neighbors_index.shape()[0],
            neighbors_row_splits.data<int64_t>(),  // NOLINT
            neighbors_row_splits.shape()[0] - 1);

    return {neighbors_index, neighbors_row_splits, neighbors_attributes};
}
#define INSTANTIATE(TIndex, TAttr)                                         \
    template std::vector<paddle::Tensor>                                   \
    InvertNeighborsListCUDA<TIndex, TAttr>(int64_t, const paddle::Tensor&, \
                                           const paddle::Tensor&,          \
                                           const paddle::Tensor&);

INSTANTIATE(int32_t, uint8_t)
INSTANTIATE(int32_t, int8_t)
INSTANTIATE(int32_t, int16_t)
INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int32_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
