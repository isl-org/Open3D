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
namespace mesh_subdivision {

/// Function to subdivide triangle mesh using the simple midpoint algorithm.
/// Each triangle is subdivided into four triangles per iteration and the
/// new vertices lie on the midpoint of the triangle edges.
/// \param number_of_iterations defines a single iteration splits each
/// triangle into four triangles that cover the same surface.
std::shared_ptr<geometry::TriangleMesh> SubdivideMidpoint(
        const geometry::TriangleMesh& mesh, int number_of_iterations);

/// Function to subdivide triangle mesh using Loop's scheme.
/// Cf. Charles T. Loop, "Smooth subdivision surfaces based on triangles",
/// 1987. Each triangle is subdivided into four triangles per iteration.
/// \param number_of_iterations defines a single iteration splits each
/// triangle into four triangles that cover the same surface.
std::shared_ptr<geometry::TriangleMesh> SubdivideLoop(
        const geometry::TriangleMesh& mesh, int number_of_iterations);

}  // namespace mesh_subdivision
}  // namespace pipelines
}  // namespace open3d
