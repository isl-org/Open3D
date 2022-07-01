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

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {

TriangleMesh TriangleMesh::CreateBox(double width,
                                     double height,
                                     double depth,
                                     core::Dtype float_dtype,
                                     core::Dtype int_dtype,
                                     const core::Device &device) {
    // Check width, height, depth.
    if (width <= 0) {
        utility::LogError("width must be > 0, but got {}", width);
    }
    if (height <= 0) {
        utility::LogError("height must be > 0, but got {}", height);
    }
    if (depth <= 0) {
        utility::LogError("depth must be > 0, but got {}", depth);
    }

    // Vertices.
    core::Tensor vertex_positions =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {width, 0.0, 0.0},
                                        {0.0, 0.0, depth},
                                        {width, 0.0, depth},
                                        {0.0, height, 0.0},
                                        {width, height, 0.0},
                                        {0.0, height, depth},
                                        {width, height, depth}},
                                       device);

    if (float_dtype == core::Float32) {
        vertex_positions = vertex_positions.To(core::Float32);
    } else if (float_dtype != core::Float64) {
        utility::LogError("float_dtype must be Float32 or Float64, but got {}.",
                          float_dtype.ToString());
    }

    // Triangles.
    core::Tensor triangle_indices = core::Tensor::Init<int64_t>({{4, 7, 5},
                                                                 {4, 6, 7},
                                                                 {0, 2, 4},
                                                                 {2, 6, 4},
                                                                 {0, 1, 2},
                                                                 {1, 3, 2},
                                                                 {1, 5, 7},
                                                                 {1, 7, 3},
                                                                 {2, 3, 7},
                                                                 {2, 7, 6},
                                                                 {0, 4, 1},
                                                                 {1, 4, 5}},
                                                                device);

    if (int_dtype == core::Int32) {
        triangle_indices = triangle_indices.To(core::Int32);
    } else if (int_dtype != core::Int64) {
        utility::LogError("int_dtype must be Int32 or Int64, but got {}.",
                          int_dtype.ToString());
    }

    // Mesh.
    TriangleMesh mesh(vertex_positions, triangle_indices);

    return mesh;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
