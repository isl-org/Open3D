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

#include "open3d/t/geometry/kernel/UVUnwrapping.h"

#include <UVAtlas.h>

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace uvunwrapping {

void ComputeUVAtlas(TriangleMesh& mesh,
                    const size_t width,
                    const size_t height,
                    const float gutter,
                    const float max_stretch,
                    float* max_stretch_out,
                    size_t* num_charts_out) {
    using namespace DirectX;
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

    HRESULT hr = UVAtlasCreate(
            pos.get(), num_verts, triangles_ptr, DXGI_FORMAT_R32_UINT,
            num_triangles, 0, max_stretch, width, height, gutter, adj.data(),
            nullptr, nullptr, nullptr, UVATLAS_DEFAULT_CALLBACK_FREQUENCY,
            UVATLAS_DEFAULT, vb, ib, nullptr, &remap, max_stretch_out,
            num_charts_out);
    if (FAILED(hr)) {
        if (hr == static_cast<HRESULT>(0x8007000DL)) {
            utility::LogError("UVAtlasCreate: Non-manifold mesh");
        } else if (hr == static_cast<HRESULT>(0x80070216L)) {
            utility::LogError("UVAtlasCreate: Arithmetic overflow");
        } else if (hr == static_cast<HRESULT>(0x80070032L)) {
            utility::LogError("UVAtlasCreate: Not supported");
        }
        utility::LogError("UVAtlasCreate failed with code 0x{:X}",
                          static_cast<uint32_t>(hr));
    }

    const uint32_t* out_triangles_ptr = reinterpret_cast<uint32_t*>(ib.data());
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
            for (int j = 0; j < 3; ++j) {
                const uint32_t& vidx = out_triangles_ptr[i * 3 + j];
                const auto& vert = vb[vidx];
                texture_uvs_ptr[i * 6 + j * 2 + 0] = vert.uv.x;
                texture_uvs_ptr[i * 6 + j * 2 + 1] = vert.uv.y;

                if (remap[vidx] != triangles_ptr[i * 3 + j]) {
                    utility::LogError(
                            "Output index buffer differs from input index "
                            "buffer.");
                }
            }
        }
    }

    texture_uvs = texture_uvs.To(mesh.GetDevice());
    mesh.SetTriangleAttr("texture_uvs", texture_uvs);
}

}  // namespace uvunwrapping
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d