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

#include "open3d/t/geometry/Grid.h"

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace t {
namespace geometry {

GridWithWeightCount::GridWithWeightCount(int resolution) {
    resolution_ = resolution;
    weights_ = std::vector<float>(resolution * resolution * resolution, 0);
    counts_ = std::vector<int>(resolution * resolution * resolution, 0);
}

void GridWithWeightCount::Insert(int x, int y, int z, float weight) {
    const int index = GetFlatIndex(x, y, z);
    weights_[index] += weight;
    counts_[index] += 1;
}

void GridWithWeightCount::InsertBatch(const core::Tensor& grid_indices,
                                      const core::Tensor& weights) {
    core::Tensor grid_indices_contiguous = grid_indices.Contiguous();
    core::Tensor weights_contiguous = weights.Contiguous();

    std::vector<int> xs = grid_indices.IndexExtract(1, 0).ToFlatVector<int>();
    std::vector<int> ys = grid_indices.IndexExtract(1, 0).ToFlatVector<int>();
    std::vector<int> zs = grid_indices.IndexExtract(1, 0).ToFlatVector<int>();
    std::vector<float> ws = weights.ToFlatVector<float>();

#pragma omp parallel for
    for (size_t i = 0; i < xs.size(); i++) {
        const int index = GetFlatIndex(xs[i], ys[i], zs[i]);
#pragma omp critical
        {
            weights_[index] += ws[i];
            counts_[index] += 1;
        }
    }
}

int GridWithWeightCount::GetFlatIndex(int x, int y, int z) const {
    return x * resolution_ * resolution_ + y * resolution_ + z;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
