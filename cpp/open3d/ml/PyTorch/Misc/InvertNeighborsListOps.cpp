#include <vector>
#include "open3d/ml/PyTorch/TorchHelper.h"
#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCPU(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes,
        int num_attributes);

#ifdef CUDA_ENABLED
template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes,
        int num_attributes);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsList(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes) {
    CHECK_CONTIGUOUS(inp_neighbors_index);
    CHECK_CONTIGUOUS(inp_neighbors_row_splits);
    CHECK_CONTIGUOUS(inp_neighbors_attributes);
    CHECK_TYPE(inp_neighbors_row_splits, kInt64);

    // check input shapes
    {
        using namespace open3d::ml::op_util;
        Dim num_neighbors("num_neighbors");

        CHECK_SHAPE(inp_neighbors_index, num_neighbors);
        CHECK_SHAPE_IGNORE_LAST_DIMS(inp_neighbors_attributes,
                                     num_neighbors || 0);
        CHECK_SHAPE(inp_neighbors_row_splits, Dim());
    }

    int num_attributes;
    if (inp_neighbors_attributes.size(0) == 0) {
        num_attributes = 0;
    } else {
        num_attributes = 1;
        for (int i = 1; i < inp_neighbors_attributes.dim(); ++i)
            num_attributes *= inp_neighbors_attributes.size(i);
    }

    const auto& index_type = inp_neighbors_index.dtype();
    const auto& attr_type = inp_neighbors_attributes.dtype();

#define FN_PARAMETERS                                          \
    num_points, inp_neighbors_index, inp_neighbors_row_splits, \
            inp_neighbors_attributes, num_attributes

#define CALL(idx_t, attr_t, fn)                  \
    if (CompareTorchDtype<idx_t>(index_type) &&  \
        CompareTorchDtype<attr_t>(attr_type)) {  \
        return fn<idx_t, attr_t>(FN_PARAMETERS); \
    }

    CHECK_SAME_DEVICE_TYPE(inp_neighbors_index, inp_neighbors_row_splits,
                           inp_neighbors_attributes);
    if (inp_neighbors_index.is_cuda()) {
#ifdef CUDA_ENABLED
        // pass to cuda function
        CALL(int32_t, int32_t, InvertNeighborsListCUDA)
        CALL(int32_t, int64_t, InvertNeighborsListCUDA)
        CALL(int32_t, float, InvertNeighborsListCUDA)
        CALL(int32_t, double, InvertNeighborsListCUDA)
#else
        TORCH_CHECK(false,
                    "InvertNeighborsList was not compiled with CUDA support")
#endif
    } else {
        CALL(int32_t, int32_t, InvertNeighborsListCPU)
        CALL(int32_t, int64_t, InvertNeighborsListCPU)
        CALL(int32_t, float, InvertNeighborsListCPU)
        CALL(int32_t, double, InvertNeighborsListCPU)
    }
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

static auto registry = torch::RegisterOperators(
        "open3d::invert_neighbors_list(int num_points, Tensor "
        "inp_neighbors_index, Tensor inp_neighbors_row_splits, Tensor "
        "inp_neighbors_attributes) -> (Tensor neighbors_index, Tensor "
        "neighbors_row_splits, Tensor neighbors_attributes)",
        &InvertNeighborsList);
