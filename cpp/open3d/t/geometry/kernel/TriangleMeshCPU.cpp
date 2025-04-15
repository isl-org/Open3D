// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstddef>

#include "open3d/core/Dtype.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/geometry/kernel/TriangleMeshImpl.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/Random.h"

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

template <typename T>
void mix_3x3(T* out, const T* a, const T* b, const T* c, float wts[3]) {
    out[0] = wts[0] * a[0] + wts[1] * b[0] + wts[2] * c[0];
    out[1] = wts[0] * a[1] + wts[1] * b[1] + wts[2] * c[1];
    out[2] = wts[0] * a[2] + wts[1] * b[2] + wts[2] * c[2];
}
template void mix_3x3<float>(float* out,
                             const float* a,
                             const float* b,
                             const float* c,
                             float wts[3]);

/// All input tensors must be on CPU, contiguous and the correct shape.
/// normals are computed if either vertex_normals and triangle_normals
/// are not empty (used in that order).
/// colors (Float32) are computed if either albedo and texture_uvs or
/// vertex_colors are not empty (used in that order).
std::array<core::Tensor, 3> SamplePointsUniformlyCPU(
        const core::Tensor& triangles,
        const core::Tensor& vertices,
        const core::Tensor& triangle_areas,
        const core::Tensor& vertex_normals,
        const core::Tensor& vertex_colors,
        const core::Tensor& triangle_normals,
        const core::Tensor& texture_uvs,
        const core::Tensor& albedo,
        size_t number_of_points) {
    utility::random::UniformRealGenerator<float> uniform_generator(0.0, 1.0);
    core::Tensor points = core::Tensor::Empty(
            {static_cast<int64_t>(number_of_points), 3}, vertices.GetDtype());
    core::Tensor normals, colors;
    bool use_vert_normal = vertex_normals.NumElements() > 0,
         use_triangle_normal = triangle_normals.NumElements() > 0,
         use_vert_colors = vertex_colors.NumElements() > 0,
         use_albedo = albedo.NumElements() > 0 && texture_uvs.NumElements() > 0;
    if (use_vert_normal || use_triangle_normal) {
        normals = core::Tensor::Empty(
                {static_cast<int64_t>(number_of_points), 3}, points.GetDtype());
    }
    if (use_albedo || use_vert_colors) {
        colors = core::Tensor::Empty(
                {static_cast<int64_t>(number_of_points), 3}, core::Float32);
    }
    const size_t tex_width = use_albedo ? albedo.GetShape(1) : 0,
                 tex_height = use_albedo ? albedo.GetShape(0) : 0;
    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(
            vertices.GetDtype(), triangles.GetDtype(), [&]() {
                const int_t* p_triangles = triangles.GetDataPtr<int_t>();
                const scalar_t* p_vertices = vertices.GetDataPtr<scalar_t>();
                const scalar_t* p_vert_normals =
                        use_vert_normal ? vertex_normals.GetDataPtr<scalar_t>()
                                        : nullptr;
                const float* p_vert_colors =
                        use_vert_colors ? vertex_colors.GetDataPtr<float>()
                                        : nullptr;
                const scalar_t* p_tri_normals =
                        use_triangle_normal
                                ? triangle_normals.GetDataPtr<scalar_t>()
                                : nullptr;
                const scalar_t* p_tri_uvs =
                        use_albedo ? texture_uvs.GetDataPtr<scalar_t>()
                                   : nullptr;
                const float* p_albedo =
                        use_albedo ? albedo.GetDataPtr<float>() : nullptr;

                scalar_t* p_points = points.GetDataPtr<scalar_t>();
                scalar_t* p_normals = (use_vert_normal || use_triangle_normal)
                                              ? normals.GetDataPtr<scalar_t>()
                                              : nullptr;
                float* p_colors = (use_albedo || use_vert_colors)
                                          ? colors.GetDataPtr<float>()
                                          : nullptr;

                utility::random::DiscreteGenerator<size_t>
                        triangle_index_generator(
                                triangle_areas.GetDataPtr<scalar_t>(),
                                triangle_areas.GetDataPtr<scalar_t>() +
                                        triangles.GetLength());
                // TODO(SS): Parallelize this.
                for (size_t point_idx = 0; point_idx < number_of_points;
                     ++point_idx, p_points += 3) {
                    float r1 = uniform_generator();
                    float r2 = uniform_generator();
                    float wts[3] = {1 - std::sqrt(r1), std::sqrt(r1) * (1 - r2),
                                    std::sqrt(r1) * r2};
                    size_t tidx = triangle_index_generator();
                    int_t vert_idx[3] = {p_triangles[3 * tidx + 0],
                                         p_triangles[3 * tidx + 1],
                                         p_triangles[3 * tidx + 2]};

                    mix_3x3(p_points, p_vertices + 3 * vert_idx[0],
                            p_vertices + 3 * vert_idx[1],
                            p_vertices + 3 * vert_idx[2], wts);

                    if (use_vert_normal) {
                        mix_3x3(p_normals, p_vert_normals + 3 * vert_idx[0],
                                p_vert_normals + 3 * vert_idx[1],
                                p_vert_normals + 3 * vert_idx[2], wts);
                        p_normals += 3;
                    } else if (use_triangle_normal) {
                        std::copy(p_tri_normals + 3 * tidx,
                                  p_tri_normals + 3 * (tidx + 1), p_normals);
                        p_normals += 3;
                    }
                    // if there is a texture, sample from texture nearest nbr
                    // pixel instead
                    if (use_albedo) {
                        float u = wts[0] * p_tri_uvs[6 * tidx] +
                                  wts[1] * p_tri_uvs[6 * tidx + 2] +
                                  wts[2] * p_tri_uvs[6 * tidx + 4];
                        float v = wts[0] * p_tri_uvs[6 * tidx + 1] +
                                  wts[1] * p_tri_uvs[6 * tidx + 3] +
                                  wts[2] * p_tri_uvs[6 * tidx + 5];
                        size_t x = u * tex_width, y = v * tex_height;
                        std::copy(p_albedo + 3 * (y * tex_width + x),
                                  p_albedo + 3 * (y * tex_width + x + 1),
                                  p_colors);
                        p_colors += 3;
                    }  // if there is no texture, sample from vertex color
                    else if (use_vert_colors) {
                        mix_3x3(p_colors, p_vert_colors + 3 * vert_idx[0],
                                p_vert_colors + 3 * vert_idx[1],
                                p_vert_colors + 3 * vert_idx[2], wts);
                        p_colors += 3;
                    }
                }
            });
    return {points, normals, colors};
}
}  // namespace trianglemesh
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
