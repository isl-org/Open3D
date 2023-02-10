// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/pytorch/ragged_tensor/RaggedTensor.h"

#include <vector>

c10::intrusive_ptr<RaggedTensor> RaggedTensor::FromRowSplits(
        torch::Tensor values, torch::Tensor row_splits, bool validate) const {
    CHECK_TYPE(row_splits, kInt64);
    values = values.contiguous();
    row_splits = row_splits.contiguous();

    if (validate) {
        TORCH_CHECK(row_splits.sizes().size() == 1,
                    "row_splits must be of rank 1");
        TORCH_CHECK(row_splits[0].item<int64_t>() == 0,
                    "Arguments to from_row_splits do not form a valid "
                    "RaggedTensor");
        for (int i = 0; i < row_splits.sizes()[0] - 1; i++) {
            if (row_splits[i].item<int64_t>() >
                row_splits[i + 1].item<int64_t>())
                TORCH_CHECK(false,
                            "row_splits must be monotonically increasing");
        }
    }

    auto device = values.device();
    row_splits = row_splits.to(device);

    return torch::make_custom_class<RaggedTensor>(values, row_splits)
            .toCustomClass<RaggedTensor>();
}

torch::Tensor RaggedTensor::GetValues() const { return _values; }
torch::Tensor RaggedTensor::GetRowSplits() const { return _row_splits; }

std::string RaggedTensor::ToString() const {
    std::ostringstream ss;
    ss << "RaggedTensor(values=" << _values.toString()
       << ", row_splits=" << _row_splits.toString() << ")";
    return ss.str();
}

torch::Tensor RaggedTensor::GetItem(int key) const {
    return _values.slice(0, _row_splits[key].item<int64_t>(),
                         _row_splits[key + 1].item<int64_t>());
}

int64_t RaggedTensor::Len() const { return _row_splits.sizes().vec()[0] - 1; }

c10::intrusive_ptr<RaggedTensor> RaggedTensor::Clone() const {
    return c10::make_intrusive<RaggedTensor>(_values.clone(), _row_splits);
}

c10::intrusive_ptr<RaggedTensor> RaggedTensor::Concat(
        c10::intrusive_ptr<RaggedTensor> r_tensor, int64_t axis) const {
    throw std::logic_error{"Function not yet Implemented."};
}
