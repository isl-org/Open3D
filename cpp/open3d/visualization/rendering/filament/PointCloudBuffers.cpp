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

#include <filament/IndexBuffer.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>
#include <Eigen/Core>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

using namespace filament;

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::float4 color = {1.0f, 1.0f, 1.0f, 1.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 1.f};
    math::float2 uv = {0.f, 0.f};

    static size_t GetPositionOffset() {
        return offsetof(ColoredVertex, position);
    }
    static size_t GetColorOffset() { return offsetof(ColoredVertex, color); }
    static size_t GetTangentOffset() {
        return offsetof(ColoredVertex, tangent);
    }
    static size_t GetUVOffset() { return offsetof(ColoredVertex, uv); }
    void SetVertexPosition(const Eigen::Vector3d& pos) {
        auto float_pos = pos.cast<float>();
        position.x = float_pos(0);
        position.y = float_pos(1);
        position.z = float_pos(2);
    }

    void SetVertexColor(const Eigen::Vector3d& c) {
        auto float_color = c.cast<float>();
        color.x = float_color(0);
        color.y = float_color(1);
        color.z = float_color(2);
    }
};
}  // namespace

PointCloudBuffersBuilder::PointCloudBuffersBuilder(
        const geometry::PointCloud& geometry)
    : geometry_(geometry) {}

RenderableManager::PrimitiveType PointCloudBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::POINTS;
}

GeometryBuffersBuilder::Buffers PointCloudBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const size_t n_vertices = geometry_.points_.size();

    // We use CUSTOM0 for tangents along with TANGENTS attribute
    // because Filament would optimize out anything about normals and lightning
    // from unlit materials. But our shader for normals visualizing is unlit, so
    // we need to use this workaround.
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(n_vertices)
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetColorOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::TANGENTS)
                                 .attribute(VertexAttribute::TANGENTS, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetTangentOffset(),
                                            sizeof(ColoredVertex))
                                 .attribute(VertexAttribute::CUSTOM0, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetTangentOffset(),
                                            sizeof(ColoredVertex))
                                 .attribute(VertexAttribute::UV0, 0,
                                            VertexBuffer::AttributeType::FLOAT2,
                                            ColoredVertex::GetUVOffset(),
                                            sizeof(ColoredVertex))
                                 .build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    math::quatf* float4v_tagents = nullptr;
    if (geometry_.HasNormals()) {
        // Converting vertex normals to float base
        std::vector<Eigen::Vector3f> normals;
        normals.resize(n_vertices);
        for (size_t i = 0; i < n_vertices; ++i) {
            normals[i] = geometry_.normals_[i].cast<float>();
        }

        // Converting normals to Filament type - quaternions
        const size_t tangents_byte_count = n_vertices * 4 * sizeof(float);
        float4v_tagents =
                static_cast<math::quatf*>(malloc(tangents_byte_count));
        auto orientation = filament::geometry::SurfaceOrientation::Builder()
                                   .vertexCount(n_vertices)
                                   .normals(reinterpret_cast<math::float3*>(
                                           normals.data()))
                                   .build();
        orientation.getQuats(float4v_tagents, n_vertices);
    }

    const size_t vertices_byte_count = n_vertices * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(vertices_byte_count));
    const ColoredVertex kDefault;
    for (size_t i = 0; i < geometry_.points_.size(); ++i) {
        ColoredVertex& element = vertices[i];
        element.SetVertexPosition(geometry_.points_[i]);
        if (geometry_.HasColors()) {
            element.SetVertexColor(geometry_.colors_[i]);
        } else {
            element.color = kDefault.color;
        }

        if (float4v_tagents) {
            element.tangent = float4v_tagents[i];
        } else {
            element.tangent = kDefault.tangent;
        }
        element.uv = kDefault.uv;
    }

    free(float4v_tagents);

    // Moving `vertices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    VertexBuffer::BufferDescriptor vb_descriptor(vertices, vertices_byte_count);
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    const size_t indices_byte_count = n_vertices * sizeof(IndexType);
    auto* uint_indices = static_cast<IndexType*>(malloc(indices_byte_count));
    for (std::uint32_t i = 0; i < n_vertices; ++i) {
        uint_indices[i] = i;
    }

    auto ib_handle =
            resource_mgr.CreateIndexBuffer(n_vertices, sizeof(IndexType));
    if (!ib_handle) {
        free(uint_indices);
        return {};
    }

    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();

    // Moving `uintIndices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    IndexBuffer::BufferDescriptor indices_descriptor(uint_indices,
                                                     indices_byte_count);
    indices_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indices_descriptor));

    return std::make_tuple(vb_handle, ib_handle);
}

filament::Box PointCloudBuffersBuilder::ComputeAABB() {
    const auto geometry_aabb = geometry_.GetAxisAlignedBoundingBox();

    const filament::math::float3 min(geometry_aabb.min_bound_.x(),
                                     geometry_aabb.min_bound_.y(),
                                     geometry_aabb.min_bound_.z());
    const filament::math::float3 max(geometry_aabb.max_bound_.x(),
                                     geometry_aabb.max_bound_.y(),
                                     geometry_aabb.max_bound_.z());

    Box aabb;
    aabb.set(min, max);

    return aabb;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
