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

#include "open3d/t/geometry/VoxelBlockGrid.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Geometry.h"

namespace open3d {
namespace t {
namespace geometry {

VoxelBlockGrid::VoxelBlockGrid(
        const std::vector<std::string> &attr_names,
        const std::vector<core::Dtype> &attr_dtypes,
        const std::vector<core::SizeVector> &attr_channels,
        double voxel_size,
        int64_t block_resolution,
        int64_t block_count,
        const core::Device &device,
        const core::HashBackendType &backend) {
    size_t n_attrs = attr_names.size();
    if (attr_dtypes.size() != n_attrs) {
        utility::LogError(
                "Number of attribute dtypes ({}) mismatch with names ({}).",
                attr_dtypes.size(), n_attrs);
    }
    if (attr_channels.size() != n_attrs) {
        utility::LogError(
                "Number of attribute channels ({}) mismatch with names ({}).",
                attr_channels.size(), n_attrs);
    }

    std::vector<core::SizeVector> attr_element_shapes;
    core::SizeVector block_shape{block_resolution, block_resolution,
                                 block_resolution};

    for (size_t i = 0; i < n_attrs; ++i) {
        // Construct element shapes.
        core::SizeVector attr_channel = attr_channels[i];
        core::SizeVector block_shape_copy = block_shape;
        block_shape_copy.insert(block_shape_copy.end(), attr_channel.begin(),
                                attr_channel.end());
        attr_element_shapes.emplace_back(block_shape_copy);

        // Used for easier accessing via attribute names.
        name_attr_map_[attr_names[i]] = i;
    }

    block_hashmap_ = std::make_shared<core::HashMap>(
            block_count, core::Int32, core::SizeVector{3}, attr_dtypes,
            attr_element_shapes, device, backend);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
