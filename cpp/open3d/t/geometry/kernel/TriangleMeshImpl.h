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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/kernel/Matrix.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace trianglemesh {

#ifndef __CUDACC__
using std::isnan;
#endif

#if defined(__CUDACC__)
void NormalizeNormalsCUDA
#else
void NormalizeNormalsCPU
#endif
        (core::Tensor& normals) {
    const core::Dtype dtype = normals.GetDtype();
    const int64_t n = normals.GetLength();

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t* ptr = normals.GetDataPtr<scalar_t>();

        core::ParallelFor(normals.GetDevice(), n,
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              int64_t idx = 3 * workload_idx;
                              scalar_t x = ptr[idx];
                              scalar_t y = ptr[idx + 1];
                              scalar_t z = ptr[idx + 2];
                              if (isnan(x)) {
                                  x = 0.0;
                                  y = 0.0;
                                  z = 1.0;
                              } else {
                                  scalar_t norm = sqrt(x * x + y * y + z * z);
                                  if (norm > 0) {
                                      x /= norm;
                                      y /= norm;
                                      z /= norm;
                                  }
                                  ptr[idx] = x;
                                  ptr[idx + 1] = y;
                                  ptr[idx + 2] = z;
                              }
                          });
    });
}

#if defined(__CUDACC__)
void ComputeTriangleNormalsCUDA
#else
void ComputeTriangleNormalsCPU
#endif
        (const core::Tensor& vertices,
         const core::Tensor& triangles,
         core::Tensor& normals) {
    const core::Dtype dtype = normals.GetDtype();
    const int64_t n = normals.GetLength();
    const core::Tensor triangles_d = triangles.To(core::Int64);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t* normal_ptr = normals.GetDataPtr<scalar_t>();
        const int64_t* triangle_ptr = triangles_d.GetDataPtr<int64_t>();
        const scalar_t* vertex_ptr = vertices.GetDataPtr<scalar_t>();

        core::ParallelFor(normals.GetDevice(), n,
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              int64_t idx = 3 * workload_idx;

                              int64_t triangle_id1 = triangle_ptr[idx];
                              int64_t triangle_id2 = triangle_ptr[idx + 1];
                              int64_t triangle_id3 = triangle_ptr[idx + 2];

                              scalar_t v01[3], v02[3];
                              v01[0] = vertex_ptr[3 * triangle_id2] -
                                       vertex_ptr[3 * triangle_id1];
                              v01[1] = vertex_ptr[3 * triangle_id2 + 1] -
                                       vertex_ptr[3 * triangle_id1 + 1];
                              v01[2] = vertex_ptr[3 * triangle_id2 + 2] -
                                       vertex_ptr[3 * triangle_id1 + 2];
                              v02[0] = vertex_ptr[3 * triangle_id3] -
                                       vertex_ptr[3 * triangle_id1];
                              v02[1] = vertex_ptr[3 * triangle_id3 + 1] -
                                       vertex_ptr[3 * triangle_id1 + 1];
                              v02[2] = vertex_ptr[3 * triangle_id3 + 2] -
                                       vertex_ptr[3 * triangle_id1 + 2];

                              core::linalg::kernel::cross_3x1(v01, v02,
                                                              &normal_ptr[idx]);
                          });
    });
}

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
