// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace mesh_sampling {

/// Function to sample \param number_of_points points uniformly from the
/// mesh. \param use_triangle_normal Set to true to assign the triangle
/// normals to the returned points instead of the interpolated vertex
/// normals. The triangle normals will be computed and added to the mesh
/// if necessary. \param seed Sets the seed value used in the random
/// generator, set to -1 to use a random seed value with each function call.
std::shared_ptr<geometry::PointCloud> SamplePointsUniformly(
        const geometry::TriangleMesh& mesh,
        size_t number_of_points,
        bool use_triangle_normal = false,
        int seed = -1);

/// Function to sample \p number_of_points points (blue noise).
/// Based on the method presented in Yuksel, "Sample Elimination for
/// Generating Poisson Disk Sample Sets", EUROGRAPHICS, 2015 The PointCloud
/// \p pcl_init is used for sample elimination if given, otherwise a
/// PointCloud is first uniformly sampled with \p init_number_of_points
/// x \p number_of_points number of points.
/// \p use_triangle_normal Set to true to assign the triangle
/// normals to the returned points instead of the interpolated vertex
/// normals. The triangle normals will be computed and added to the mesh
/// if necessary. \p seed Sets the seed value used in the random
/// generator, set to -1 to use a random seed value with each function call.
std::shared_ptr<geometry::PointCloud> SamplePointsPoissonDisk(
        const geometry::TriangleMesh& mesh,
        size_t number_of_points,
        double init_factor = 5,
        const std::shared_ptr<geometry::PointCloud> pcl_init = nullptr,
        bool use_triangle_normal = false,
        int seed = -1);

}  // namespace mesh_sampling
}  // namespace pipelines
}  // namespace open3d
