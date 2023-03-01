// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/UVUnwrapping.h"

// clang-format off
// include tbb before uvatlas
#include <tbb/parallel_for.h>
#include <UVAtlas.h>
// clang-format on

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace uvunwrapping {

namespace {
using namespace DirectX;

struct UVAtlasPartitionOutput {
    std::vector<int64_t> original_face_idx;
    std::vector<UVAtlasVertex> vb;
    // raw index buffer. Elements are uint32_t for DXGI_FORMAT_R32_UINT.
    std::vector<uint8_t> ib;
    std::vector<uint32_t> partition_result_adjacency;
    float max_stretch_out;
    size_t num_charts_out;
};

void ComputeUVAtlasPartition(TriangleMesh mesh,
                             const float max_stretch,
                             bool fast_mode,
                             UVAtlasPartitionOutput& output) {
    const int64_t num_verts = mesh.GetVertexPositions().GetLength();

    std::unique_ptr<XMFLOAT3[]> pos(new XMFLOAT3[num_verts]);
    {
        core::Tensor vertices = mesh.GetVertexPositions()
                                        .To(core::Device(), core::Float32)
                                        .Contiguous();
        const float* vertices_ptr = vertices.GetDataPtr<float>();
        for (int64_t i = 0; i < num_verts; ++i) {
            pos[i].x = vertices_ptr[i * 3 + 0];
            pos[i].y = vertices_ptr[i * 3 + 1];
            pos[i].z = vertices_ptr[i * 3 + 2];
        }
    }

    core::Tensor triangles = mesh.GetTriangleIndices()
                                     .To(core::Device(), core::UInt32)
                                     .Contiguous();
    const uint32_t* triangles_ptr = triangles.GetDataPtr<uint32_t>();
    const int64_t num_triangles = triangles.GetLength();

    typedef uint64_t Edge_t;
    typedef std::pair<uint32_t, uint32_t> AdjTriangles_t;
    const uint32_t INVALID = static_cast<uint32_t>(-1);
    auto MakeEdge = [](uint32_t idx1, uint32_t idx2) {
        return (uint64_t(std::min(idx1, idx2)) << 32) | std::max(idx1, idx2);
    };

    // Compute adjacency as described here
    // https://github.com/microsoft/DirectXMesh/wiki/DirectXMesh
    std::vector<uint32_t> adj(triangles.NumElements());
    {
        std::unordered_map<Edge_t, AdjTriangles_t> edge_adjtriangle_map;
        for (int64_t i = 0; i < num_triangles; ++i) {
            const uint32_t* t_ptr = triangles_ptr + i * 3;

            for (int j = 0; j < 3; ++j) {
                auto e = MakeEdge(t_ptr[j], t_ptr[(j + 1) % 3]);
                auto it = edge_adjtriangle_map.find(e);
                if (it != edge_adjtriangle_map.end()) {
                    it->second.second = i;
                } else {
                    edge_adjtriangle_map[e] = AdjTriangles_t(i, INVALID);
                }
            }
        }

        // second pass filling the adj array
        int64_t linear_idx = 0;
        for (int64_t i = 0; i < num_triangles; ++i) {
            const uint32_t* t_ptr = triangles_ptr + i * 3;
            for (int j = 0; j < 3; ++j, ++linear_idx) {
                auto e = MakeEdge(t_ptr[j], t_ptr[(j + 1) % 3]);
                auto& adjacent_tri = edge_adjtriangle_map[e];
                if (adjacent_tri.first != i) {
                    adj[linear_idx] = adjacent_tri.first;
                } else {
                    adj[linear_idx] = adjacent_tri.second;
                }
            }
        }
    }

    // Output vertex buffer for positions and uv coordinates.
    // Note that the positions will be modified during the atlas computation
    // and don't represent the original mesh anymore.
    std::vector<UVAtlasVertex> vb;
    // raw index buffer. Elements are uint32_t for DXGI_FORMAT_R32_UINT.
    std::vector<uint8_t> ib;
    // UVAtlas will create new vertices. remap stores the original vertex id for
    // all created vertices.
    std::vector<uint32_t> remap;
    std::vector<uint32_t> face_partitioning;
    std::vector<uint32_t> partition_result_adjacency;

    HRESULT hr = UVAtlasPartition(
            pos.get(), num_verts, triangles_ptr, DXGI_FORMAT_R32_UINT,
            num_triangles, 0, max_stretch, adj.data(), nullptr, nullptr,
            nullptr, UVATLAS_DEFAULT_CALLBACK_FREQUENCY,
            fast_mode ? UVATLAS_GEODESIC_FAST : UVATLAS_DEFAULT, vb, ib,
            &face_partitioning, &remap, partition_result_adjacency,
            &output.max_stretch_out, &output.num_charts_out);

    if (FAILED(hr)) {
        if (hr == static_cast<HRESULT>(0x8007000DL)) {
            utility::LogError("UVAtlasPartition: Non-manifold mesh");
        } else if (hr == static_cast<HRESULT>(0x80070216L)) {
            utility::LogError("UVAtlasPartition: Arithmetic overflow");
        } else if (hr == static_cast<HRESULT>(0x80070032L)) {
            utility::LogError("UVAtlasPartition: Not supported");
        }
        utility::LogError("UVAtlasPartition failed with code 0x{:X}",
                          static_cast<uint32_t>(hr));
    }

    output.original_face_idx =
            mesh.GetTriangleAttr("original_idx").ToFlatVector<int64_t>();
    output.ib = std::move(ib);
    output.vb = std::move(vb);
    output.partition_result_adjacency = std::move(partition_result_adjacency);
}
}  // namespace

std::tuple<float, int, int> ComputeUVAtlas(TriangleMesh& mesh,
                                           const size_t width,
                                           const size_t height,
                                           const float gutter,
                                           const float max_stretch,
                                           int parallel_partitions,
                                           int nthreads) {
    const int64_t num_verts = mesh.GetVertexPositions().GetLength();

    // create temporary mesh for partitioning
    TriangleMesh mesh_tmp(
            mesh.GetVertexPositions().To(core::Device()).Contiguous(),
            mesh.GetTriangleIndices().To(core::Device()).Contiguous());
    if (parallel_partitions > 1) {
        const int max_points_per_partition =
                (num_verts - 1) / (parallel_partitions - 1);
        parallel_partitions = mesh_tmp.PCAPartition(max_points_per_partition);
    }
    utility::LogInfo("actual parallel_partitions {}", parallel_partitions);
    mesh_tmp.SetTriangleAttr(
            "original_idx",
            core::Tensor::Arange(0, mesh_tmp.GetTriangleIndices().GetLength(),
                                 1, core::Int64));

    std::vector<TriangleMesh> mesh_partitions;
    if (parallel_partitions > 1) {
        for (int i = 0; i < parallel_partitions; ++i) {
            core::Tensor mask = mesh_tmp.GetTriangleAttr("partition_ids").Eq(i);
            mesh_partitions.emplace_back(mesh_tmp.SelectFacesByMask(mask));
        }
    } else {
        mesh_partitions.emplace_back(mesh_tmp);
    }

    std::vector<UVAtlasPartitionOutput> uvatlas_partitions(parallel_partitions);

    // By default UVAtlas uses a fast mode if there are more than 25k faces.
    // This makes sure that we always use fast mode if we parallelize.
    const bool fast_mode = parallel_partitions > 1;
    auto LoopFn = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i < range.end(); ++i) {
            auto& output = uvatlas_partitions[i];
            ComputeUVAtlasPartition(mesh_partitions[i], max_stretch, fast_mode,
                                    output);
        }
    };

    if (parallel_partitions > 1) {
        if (nthreads > 0) {
            tbb::task_arena arena(nthreads);
            arena.execute([&]() {
                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, parallel_partitions),
                        LoopFn);
            });
        } else {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, parallel_partitions), LoopFn);
        }
    } else {
        LoopFn(tbb::blocked_range<size_t>(0, parallel_partitions));
    }

    // merge outputs for the packing algorithm
    UVAtlasPartitionOutput& combined_output = uvatlas_partitions.front();
    for (int i = 1; i < parallel_partitions; ++i) {
        auto& output = uvatlas_partitions[i];

        // append vectors and update indices
        const uint32_t vidx_offset = combined_output.vb.size();
        uint32_t* indices_ptr = reinterpret_cast<uint32_t*>(output.ib.data());
        const int64_t num_indices = output.ib.size() / sizeof(uint32_t);
        for (int64_t indices_i = 0; indices_i < num_indices; ++indices_i) {
            indices_ptr[indices_i] += vidx_offset;
        }
        combined_output.vb.insert(combined_output.vb.end(), output.vb.begin(),
                                  output.vb.end());

        const uint32_t fidx_offset =
                combined_output.ib.size() / (sizeof(uint32_t) * 3);
        const uint32_t invalid = std::numeric_limits<uint32_t>::max();
        for (auto& x : output.partition_result_adjacency) {
            if (x != invalid) {
                x += fidx_offset;
            }
        }
        combined_output.ib.insert(combined_output.ib.end(), output.ib.begin(),
                                  output.ib.end());
        combined_output.partition_result_adjacency.insert(
                combined_output.partition_result_adjacency.end(),
                output.partition_result_adjacency.begin(),
                output.partition_result_adjacency.end());

        combined_output.original_face_idx.insert(
                combined_output.original_face_idx.end(),
                output.original_face_idx.begin(),
                output.original_face_idx.end());

        // update stats
        combined_output.max_stretch_out = std::max(
                combined_output.max_stretch_out, output.max_stretch_out);
        combined_output.num_charts_out += output.num_charts_out;

        // free memory
        output = UVAtlasPartitionOutput();
    }

    HRESULT hr = UVAtlasPack(combined_output.vb, combined_output.ib,
                             DXGI_FORMAT_R32_UINT, width, height, gutter,
                             combined_output.partition_result_adjacency,
                             nullptr, UVATLAS_DEFAULT_CALLBACK_FREQUENCY);

    if (FAILED(hr)) {
        if (hr == static_cast<HRESULT>(0x8007000DL)) {
            utility::LogError("UVAtlasPack: Non-manifold mesh");
        } else if (hr == static_cast<HRESULT>(0x80070216L)) {
            utility::LogError("UVAtlasPack: Arithmetic overflow");
        } else if (hr == static_cast<HRESULT>(0x80070032L)) {
            utility::LogError("UVAtlasPack: Not supported");
        }
        utility::LogError("UVAtlasPack failed with code 0x{:X}",
                          static_cast<uint32_t>(hr));
    }

    auto& ib = combined_output.ib;
    auto& vb = combined_output.vb;
    auto& original_face_idx = combined_output.original_face_idx;
    const int64_t num_triangles = mesh.GetTriangleIndices().GetLength();
    const uint32_t* indices_ptr = reinterpret_cast<uint32_t*>(ib.data());
    if (ib.size() != sizeof(uint32_t) * 3 * num_triangles) {
        utility::LogError(
                "Unexpected output index buffer size. Got {} expected {}.",
                ib.size(), sizeof(uint32_t) * 3 * num_triangles);
    }
    core::Tensor texture_uvs({num_triangles, 3, 2}, core::Float32);
    {
        // copy uv coords
        float* texture_uvs_ptr = texture_uvs.GetDataPtr<float>();
        for (int64_t i = 0; i < num_triangles; ++i) {
            int64_t original_i = original_face_idx[i];
            for (int j = 0; j < 3; ++j) {
                const uint32_t& vidx = indices_ptr[i * 3 + j];
                const auto& vert = vb[vidx];
                texture_uvs_ptr[original_i * 6 + j * 2 + 0] = vert.uv.x;
                texture_uvs_ptr[original_i * 6 + j * 2 + 1] = vert.uv.y;
            }
        }
    }

    texture_uvs = texture_uvs.To(mesh.GetDevice());
    mesh.SetTriangleAttr("texture_uvs", texture_uvs);

    return std::tie(combined_output.max_stretch_out,
                    combined_output.num_charts_out, parallel_partitions);
}

}  // namespace uvunwrapping
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d