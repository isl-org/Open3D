// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/TriangleMeshImpl.h"

#include <sycl/sycl.hpp>

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
        core::ParallelFor(
                vertex_normals.GetDevice(), n,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t idx = 3 * workload_idx;
                    int64_t triangle_id1 = triangle_ptr[idx];
                    int64_t triangle_id2 = triangle_ptr[idx + 1];
                    int64_t triangle_id3 = triangle_ptr[idx + 2];

                    scalar_t n1 = triangle_normals_ptr[idx];
                    scalar_t n2 = triangle_normals_ptr[idx + 1];
                    scalar_t n3 = triangle_normals_ptr[idx + 2];

                    auto v1_0 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id1]);
                    v1_0.fetch_add(n1);
                    auto v1_1 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id1 + 1]);
                    v1_1.fetch_add(n2);
                    auto v1_2 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id1 + 2]);
                    v1_2.fetch_add(n3);

                    auto v2_0 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id2]);
                    v2_0.fetch_add(n1);
                    auto v2_1 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id2 + 1]);
                    v2_1.fetch_add(n2);
                    auto v2_2 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id2 + 2]);
                    v2_2.fetch_add(n3);

                    auto v3_0 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id3]);
                    v3_0.fetch_add(n1);
                    auto v3_1 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id3 + 1]);
                    v3_1.fetch_add(n2);
                    auto v3_2 = sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>(
                            vertex_normals_ptr[3 * triangle_id3 + 2]);
                    v3_2.fetch_add(n3);
                });
    });
}

}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
