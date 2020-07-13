#include <ATen/cuda/CUDAContext.h>
#include "open3d/ml/PyTorch/TorchHelper.h"
#include "open3d/ml/impl/misc/InvertNeighborsList.cuh"
#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes,
        int num_attributes) {
    auto device = inp_neighbors_index.device().type();
    auto device_idx = inp_neighbors_index.device().index();
    torch::Tensor neighbors_index = torch::empty(
            inp_neighbors_index.sizes(),
            torch::dtype(ToTorchDtype<TIndex>()).device(device, device_idx));
    torch::Tensor neighbors_row_splits = torch::empty(
            {num_points + 1},
            torch::dtype(torch::kInt64).device(device, device_idx));
    torch::Tensor neighbors_attributes =
            torch::empty_like(inp_neighbors_attributes);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    // determine temp_size
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes,
            (int64_t*)inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0),
            (int64_t*)neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_row_splits.size(0) - 1);

    torch::Tensor temp_tensor = torch::empty(
            {int64_t(temp_size)},
            torch::dtype(ToTorchDtype<uint8_t>()).device(device, device_idx));
    temp_ptr = temp_tensor.data_ptr<uint8_t>();

    // actually invert the list
    open3d::ml::impl::InvertNeighborsListCUDA(
            stream, temp_ptr, temp_size, texture_alignment,
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes,
            (int64_t*)inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0),
            (int64_t*)neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_row_splits.size(0) - 1);

    return std::make_tuple(neighbors_index, neighbors_row_splits,
                           neighbors_attributes);
}
#define INSTANTIATE(TIndex, TAttr)                                       \
    template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>     \
    InvertNeighborsListCUDA<TIndex, TAttr>(int64_t, torch::Tensor,       \
                                           torch::Tensor, torch::Tensor, \
                                           int num_attributes);

INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int32_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
