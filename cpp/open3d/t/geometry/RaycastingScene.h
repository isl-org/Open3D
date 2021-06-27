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

#pragma once

#include <memory>

#include "open3d/Macro.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {

/// A scene which provides basic ray casting and closest point queries.
class RaycastingScene {
public:
    /// \brief Default Constructor.
    RaycastingScene();

    ~RaycastingScene();

    uint32_t AddTriangles(const core::Tensor &vertices,
                          const core::Tensor &triangles);

    uint32_t AddTriangles(const TriangleMesh &mesh);

    std::unordered_map<std::string, core::Tensor> CastRays(
            const core::Tensor &rays);

    core::Tensor CountIntersections(const core::Tensor &rays);

    std::unordered_map<std::string, core::Tensor> ComputeClosestPoints(
            const core::Tensor &query_points);

    core::Tensor ComputeDistance(const core::Tensor &query_points);

    core::Tensor ComputeSignedDistance(const core::Tensor &query_points,
                                       bool use_triangle_normal = false);

    core::Tensor ComputeOccupancy(const core::Tensor &query_points,
                                  bool use_triangle_normal = false);

    static uint32_t INVALID_ID();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
