#include "open3d/ml/PyTorch/TorchHelper.h"
#include "open3d/ml/impl/misc/InvertNeighborsList.h"
#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCPU(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes,
        int num_attributes) {
    torch::Tensor neighbors_index = torch::empty(
            inp_neighbors_index.sizes(), torch::dtype(ToTorchDtype<TIndex>()));
    torch::Tensor neighbors_row_splits =
            torch::empty({num_points + 1}, torch::dtype(torch::kInt64));
    torch::Tensor neighbors_attributes =
            torch::empty_like(inp_neighbors_attributes);

    open3d::ml::impl::InvertNeighborsListCPU(
            inp_neighbors_index.data_ptr<TIndex>(),
            num_attributes ? inp_neighbors_attributes.data_ptr<TAttr>()
                           : nullptr,
            num_attributes, inp_neighbors_row_splits.data_ptr<int64_t>(),
            inp_neighbors_row_splits.size(0) - 1,
            neighbors_index.data_ptr<TIndex>(),
            num_attributes ? neighbors_attributes.data_ptr<TAttr>() : nullptr,
            neighbors_index.size(0), neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_row_splits.size(0) - 1);

    return std::make_tuple(neighbors_index, neighbors_row_splits,
                           neighbors_attributes);
}
#define INSTANTIATE(TIndex, TAttr)                                      \
    template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>    \
    InvertNeighborsListCPU<TIndex, TAttr>(int64_t, torch::Tensor,       \
                                          torch::Tensor, torch::Tensor, \
                                          int num_attributes);

INSTANTIATE(int32_t, int32_t)
INSTANTIATE(int32_t, int64_t)
INSTANTIATE(int32_t, float)
INSTANTIATE(int32_t, double)
