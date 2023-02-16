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

#include "open3d/t/geometry/kernel/TriangleMeshImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace trianglemesh {

void ComputeVertexNormalsCPU(const core::Tensor& triangles,
                             const core::Tensor& triangle_normals,
                             core::Tensor& vertex_normals) {
    const core::Dtype dtype = vertex_normals.GetDtype();
    const int64_t n = triangles.GetLength();
    const core::Tensor triangles_d = triangles.To(core::Int64);

    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const int64_t* triangle_ptr = triangles_d.GetDataPtr<int64_t>();
        const scalar_t* triangle_normals_ptr =
                triangle_normals.GetDataPtr<scalar_t>();
        scalar_t* vertex_normals_ptr = vertex_normals.GetDataPtr<scalar_t>();

        for (int64_t i = 0; i < n; ++i) {
            int64_t idx = 3 * i;
            int64_t triangle_id1 = triangle_ptr[idx];
            int64_t triangle_id2 = triangle_ptr[idx + 1];
            int64_t triangle_id3 = triangle_ptr[idx + 2];

            scalar_t n1 = triangle_normals_ptr[idx];
            scalar_t n2 = triangle_normals_ptr[idx + 1];
            scalar_t n3 = triangle_normals_ptr[idx + 2];

            vertex_normals_ptr[3 * triangle_id1] += n1;
            vertex_normals_ptr[3 * triangle_id1 + 1] += n2;
            vertex_normals_ptr[3 * triangle_id1 + 2] += n3;
            vertex_normals_ptr[3 * triangle_id2] += n1;
            vertex_normals_ptr[3 * triangle_id2 + 1] += n2;
            vertex_normals_ptr[3 * triangle_id2 + 2] += n3;
            vertex_normals_ptr[3 * triangle_id3] += n1;
            vertex_normals_ptr[3 * triangle_id3 + 1] += n2;
            vertex_normals_ptr[3 * triangle_id3 + 2] += n3;
        }
    });
}

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
