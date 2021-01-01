// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/t/pipelines/slac/ControlGrid.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {
ControlGrid::ControlGrid(float grid_size,
                         int64_t grid_count,
                         const core::Device& device)
    : grid_size_(grid_size), device_(device) {
    // Maps a coordinate to its non-rigidly transformed coordinate
    ctr_hashmap_ = std::make_shared<core::Hashmap>(
            grid_count, core::Dtype::Int32, core::Dtype::Float32,
            core::SizeVector{3}, core::SizeVector{3}, device);
}

void ControlGrid::Touch(const geometry::PointCloud& pcd) {
    core::Tensor pts = pcd.GetPoints();
    int64_t n = pts.GetLength();

    core::Tensor vals = (pts / grid_size_).Floor();
    core::Tensor keys = vals.To(core::Dtype::Int32);

    core::Tensor keys_nb({8, n}, core::Dtype::Int32, device_);
    core::Tensor vals_nb({8, n, 3}, core::Dtype::Int32, device_);
    for (int nb = 0; nb < 8; ++nb) {
        core::Tensor dt =
                core::Tensor(std::vector<int>{(nb & 4), (nb & 2), (nb & 1)},
                             {1, 3}, core::Dtype::Int32, device_);
        keys_nb[nb] = keys + dt;
        vals_nb[nb] = vals + dt;
    }
    keys_nb = keys_nb.View({8 * n, 1});
    vals_nb = vals_nb.View({8 * n, 3}) * grid_size_;

    core::Tensor addrs_nb, masks_nb;
    ctr_hashmap_->Insert(keys_nb, vals_nb, addrs_nb, masks_nb);
}
}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
