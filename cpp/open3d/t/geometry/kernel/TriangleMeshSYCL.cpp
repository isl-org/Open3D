// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file TriangleMeshSYCL.cpp
/// \brief SYCL triangle-mesh kernels (see TriangleMeshImpl.h).

#include "open3d/core/ParallelFor.h"
#include "open3d/t/geometry/kernel/TriangleMeshImpl.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace trianglemesh {

void ComputeVertexNormalsSYCL(const core::Tensor& triangles,
                              const core::Tensor& triangle_normals,
                              core::Tensor& vertex_normals) {
    const core::Dtype dtype = vertex_normals.GetDtype();
    const int64_t n = triangles.GetLength();
    const core::Tensor triangles_d = triangles.To(core::Int64);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const int64_t* triangle_ptr = triangles_d.GetDataPtr<int64_t>();
        const scalar_t* triangle_normals_ptr =
                triangle_normals.GetDataPtr<scalar_t>();
        scalar_t* vertex_normals_ptr = vertex_normals.GetDataPtr<scalar_t>();
        core::ParallelFor(vertex_normals.GetDevice(), n,
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              int64_t idx = 3 * workload_idx;
                              const int64_t vertex_ids[3] = {
                                      triangle_ptr[idx], triangle_ptr[idx + 1],
                                      triangle_ptr[idx + 2]};
                              const scalar_t n1 = triangle_normals_ptr[idx];
                              const scalar_t n2 = triangle_normals_ptr[idx + 1];
                              const scalar_t n3 = triangle_normals_ptr[idx + 2];

                              for (int vi = 0; vi < 3; ++vi) {
                                  const int64_t base = 3 * vertex_ids[vi];
                                  OPEN3D_ATOMIC_ADD_RELAXED(
                                          &vertex_normals_ptr[base], n1);
                                  OPEN3D_ATOMIC_ADD_RELAXED(
                                          &vertex_normals_ptr[base + 1], n2);
                                  OPEN3D_ATOMIC_ADD_RELAXED(
                                          &vertex_normals_ptr[base + 2], n3);
                              }
                          });
    });
}

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
