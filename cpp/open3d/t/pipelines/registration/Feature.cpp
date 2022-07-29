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

#include "open3d/t/pipelines/registration/Feature.h"

#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/t/geometry/PointCloud.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace registration {

core::Tensor ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const int max_nn = 100,
        const utility::optional<double> radius = utility::nullopt) {
    core::AssertTensorDtypes(input.GetPointPositions(),
                             {core::Float64, core::Float32});
    if (max_nn <= 1) {
        utility::LogError("max_nn must be greater than 1.");
    }
    if (radius.value() <= 0) {
        utility::LogError("radius must be greater than 0.");
    }

    // Compute nearest neighbors and squared distances.
    core::Tensor indices, distance2;
    core::nns::NearestNeighborSearch tree(input.GetPointPositions(),
                                          core::Int32);
    if (radius.has_value()) {
        bool check = tree.HybridIndex(radius.value());
        if (!check) {
            utility::LogError("Building HybridIndex failed.");
        }
        core::Tensor counts;
        std::tie(indices, distance2, counts) = tree.HybridSearch(
                input.GetPointPositions(), radius.value(), max_nn);
    } else {
        bool check = tree.KnnIndex();
        if (!check) {
            utility::LogError("Building KnnIndex failed.");
        }
        std::tie(indices, distance2) = tree.KnnSearch(
                input.GetPointPositions(), max_nn);
    }

    const core::Device device = input.GetDevice();
    const core::Dtype dtype = input.GetPointPositions().GetDtype();
    const int64_t size = input.GetPointPositions().GetLength();

    core::Tensor fpfh = core::Tensor::Zeros({size, 33}, dtype, device);

}

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
