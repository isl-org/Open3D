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

#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace mesh_simplification {
/// \brief Indicates the method that is used for mesh simplification if
/// multiple vertices are combined to a single one.
///
/// \param Average indicates that the average position is computed as
/// output.
/// \param Quadric indicates that the distance to the adjacent triangle
/// planes is minimized. Cf. "Simplifying Surfaces with Color and Texture
/// using Quadric Error Metrics" by Garland and Heckbert.
enum class SimplificationContraction { Average, Quadric };

/// Function to simplify mesh using Vertex Clustering.
/// The result can be a non-manifold mesh.
/// \param voxel_size - The size of the voxel within vertices are pooled.
/// \param contraction - Method to aggregate vertex information. Average
/// computes a simple average, Quadric minimizes the distance to the
/// adjacent planes.
std::shared_ptr<geometry::TriangleMesh> SimplifyVertexClustering(
        const geometry::TriangleMesh& mesh,
        double voxel_size,
        SimplificationContraction contraction =
                SimplificationContraction::Average);

/// Function to simplify mesh using Quadric Error Metric Decimation by
/// Garland and Heckbert.
/// \param target_number_of_triangles defines the number of triangles that
/// the simplified mesh should have. It is not guaranteed that this number
/// will be reached.
std::shared_ptr<geometry::TriangleMesh> SimplifyQuadricDecimation(
        const geometry::TriangleMesh& mesh, int target_number_of_triangles);

}  // namespace mesh_simplification
}  // namespace pipelines
}  // namespace open3d
