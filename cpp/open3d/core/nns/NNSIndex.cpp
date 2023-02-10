// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/NNSIndex.h"

namespace open3d {
namespace core {
namespace nns {

int NNSIndex::GetDimension() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

size_t NNSIndex::GetDatasetSize() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

Dtype NNSIndex::GetDtype() const { return dataset_points_.GetDtype(); }

Device NNSIndex::GetDevice() const { return dataset_points_.GetDevice(); }

Dtype NNSIndex::GetIndexDtype() const { return index_dtype_; }

}  // namespace nns
}  // namespace core
}  // namespace open3d
