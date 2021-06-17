// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 - 2021 www.open3d.org
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

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif  // _MSC_VER

#include <filament/IndexBuffer.h>
#include <filament/VertexBuffer.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <map>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/LineSet.h"
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
    math::float4 next = {0.f, 0.f, 0.f, 1.f};
    math::float4 color = {1.f, 1.f, 1.f, 1.f};

    static std::uint32_t GetPositionOffset() {
        return offsetof(ColoredVertex, position);
    }
    static std::uint32_t GetNextOffset() {
        return offsetof(ColoredVertex, next);
    }
    static std::uint32_t GetColorOffset() {
        return offsetof(ColoredVertex, color);
    }

    void SetVertexPosition(const Eigen::Vector3d& pos) {
        auto float_pos = pos.cast<float>();
        position.x = float_pos(0);
        position.y = float_pos(1);
        position.z = float_pos(2);
    }

    void SetVertexNext(const Eigen::Vector3d& pos, float dir) {
        auto float_pos = pos.cast<float>();
        next.x = float_pos(0);
        next.y = float_pos(1);
        next.z = float_pos(2);
        next.w = dir;
    }

    void SetVertexColor(const Eigen::Vector3d& c) {
        auto float_color = c.cast<float>();
        color.x = float_color(0);
        color.y = float_color(1);
        color.z = float_color(2);
        color.w = 1.0f;
    }
};

}  // namespace

LineSetBuffersBuilder::LineSetBuffersBuilder(const geometry::LineSet& geometry)
    : geometry_(geometry) {}

RenderableManager::PrimitiveType LineSetBuffersBuilder::GetPrimitiveType()
        const {
    if (wide_lines_) {
        return RenderableManager::PrimitiveType::TRIANGLES;
    } else {
        return RenderableManager::PrimitiveType::LINES;
    }
}

LineSetBuffersBuilder::Buffers LineSetBuffersBuilder::ConstructThinLines() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    struct LookupKey {
        LookupKey() = default;
        explicit LookupKey(const Eigen::Vector3d& pos,
                           const Eigen::Vector3d& color) {
            values[0] = pos.x();
            values[1] = pos.y();
            values[2] = pos.z();
            values[3] = color.x();
            values[4] = color.y();
            values[5] = color.z();
        }

        // Not necessarily transitive.
        // TODO: does this break sort and map?
        bool operator<(const LookupKey& other) const {
            for (int i = 0; i < 6; ++i) {
                double diff = abs(values[i] - other.values[i]);
                if (diff > kEpsilon) {
                    return values[i] < other.values[i];
                }
            }

            return false;
        }

        const double kEpsilon = 0.00001;
        double values[6] = {0};
    };

    // <source, real>
    std::map<LookupKey, std::pair<GeometryBuffersBuilder::IndexType,
                                  GeometryBuffersBuilder::IndexType>>
            index_lookup;

    const size_t lines_count = geometry_.lines_.size();
    const size_t vertices_bytes_count = lines_count * 2 * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(vertices_bytes_count));

    const size_t indices_bytes_count = lines_count * 2 * sizeof(IndexType);
    auto* indices = static_cast<IndexType*>(malloc(indices_bytes_count));

    const bool has_colors = geometry_.HasColors();
    Eigen::Vector3d kWhite(1.0, 1.0, 1.0);
    size_t vertex_idx = 0;
    for (size_t i = 0; i < lines_count; ++i) {
        const auto& line = geometry_.lines_[i];

        for (size_t j = 0; j < 2; ++j) {
            size_t index = line(j);

            auto& color = kWhite;
            if (has_colors) {
                color = geometry_.colors_[i];
            }
            const auto& pos = geometry_.points_[index];
            LookupKey lookup_key(pos, color);
            auto found = index_lookup.find(lookup_key);
            if (found != index_lookup.end()) {
                index = found->second.second;
            } else {
                auto& element = vertices[vertex_idx];

                element.SetVertexPosition(pos);
                element.SetVertexColor(color);

                index_lookup[lookup_key] = {IndexType(index),
                                            IndexType(vertex_idx)};
                index = vertex_idx;

                ++vertex_idx;
            }

            indices[2 * i + j] = IndexType(index);
        }
    }

    const size_t vertices_count = vertex_idx;

    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(std::uint32_t(vertices_count))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetColorOffset(),
                                            sizeof(ColoredVertex))
                                 .build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        free(vertices);
        free(indices);
        return {};
    }

    // Moving `vertices` to VertexBuffer, which will clean them up later
    // with DeallocateBuffer
    VertexBuffer::BufferDescriptor vb_descriptor(
            vertices, vertices_count * sizeof(ColoredVertex));
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    const size_t indices_count = lines_count * 2;
    auto ib_handle =
            resource_mgr.CreateIndexBuffer(indices_count, sizeof(IndexType));
    if (!ib_handle) {
        free(indices);
        return {};
    }

    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();

    // Moving `indices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    IndexBuffer::BufferDescriptor ib_descriptor(indices, indices_bytes_count);
    ib_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(ib_descriptor));

    return std::make_tuple(vb_handle, ib_handle, IndexBufferHandle());
}

LineSetBuffersBuilder::Buffers LineSetBuffersBuilder::ConstructBuffers() {
    // Build lines instead of triangles unless wide lines are specified
    if (!wide_lines_) {
        return ConstructThinLines();
    }

    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const size_t lines_count = geometry_.lines_.size();
    // NOTE: Vertices are duplicated so you need double (x4 instead of x2) the
    // bytes
    const size_t vertices_bytes_count = lines_count * 4 * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(vertices_bytes_count));

    // NOTE: Each line is 2 triangles
    const size_t indices_bytes_count = lines_count * 6 * sizeof(IndexType);
    auto* indices = static_cast<IndexType*>(malloc(indices_bytes_count));

    const bool has_colors = geometry_.HasColors();
    Eigen::Vector3d kWhite(1.0, 1.0, 1.0);
    size_t vertex_idx = 0;
    size_t index_idx = 0;
    for (size_t i = 0; i < lines_count; ++i) {
        const auto& line = geometry_.lines_[i];

        auto& color = kWhite;
        if (has_colors) {
            color = geometry_.colors_[i];
        }

        const auto& pos1 = geometry_.points_[line(0)];
        const auto& pos2 = geometry_.points_[line(1)];

        auto& element1 = vertices[vertex_idx];
        element1.SetVertexPosition(pos1);
        element1.SetVertexNext(pos2, 1.f);
        element1.SetVertexColor(color);

        auto& element2 = vertices[vertex_idx + 1];
        element2.SetVertexPosition(pos1);
        element2.SetVertexNext(pos2, -1.f);
        element2.SetVertexColor(color);

        auto& element3 = vertices[vertex_idx + 2];
        element3.SetVertexPosition(pos2);
        element3.SetVertexNext(pos1, -1.f);
        element3.SetVertexColor(color);

        auto& element4 = vertices[vertex_idx + 3];
        element4.SetVertexPosition(pos2);
        element4.SetVertexNext(pos1, 1.f);
        element4.SetVertexColor(color);

        // Triangle 1
        indices[index_idx++] = IndexType(vertex_idx);
        indices[index_idx++] = IndexType(vertex_idx + 1);
        indices[index_idx++] = IndexType(vertex_idx + 2);

        // Triangle 2
        indices[index_idx++] = IndexType(vertex_idx + 3);
        indices[index_idx++] = IndexType(vertex_idx + 2);
        indices[index_idx++] = IndexType(vertex_idx + 1);

        vertex_idx += 4;
    }

    const size_t vertices_count = vertex_idx;

    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(std::uint32_t(vertices_count))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
                                            sizeof(ColoredVertex))
                                 .attribute(VertexAttribute::CUSTOM0, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetNextOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetColorOffset(),
                                            sizeof(ColoredVertex))
                                 .build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        free(vertices);
        free(indices);
        return {};
    }

    // Moving `vertices` to VertexBuffer, which will clean them up later
    // with DeallocateBuffer
    VertexBuffer::BufferDescriptor vb_descriptor(
            vertices, vertices_count * sizeof(ColoredVertex));
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    // const size_t indices_count = lines_count * 6;
    const size_t indices_count = index_idx;
    auto ib_handle =
            resource_mgr.CreateIndexBuffer(indices_count, sizeof(IndexType));
    if (!ib_handle) {
        free(indices);
        return {};
    }

    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();

    // Moving `indices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    IndexBuffer::BufferDescriptor ib_descriptor(indices, indices_bytes_count);
    ib_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(ib_descriptor));

    return std::make_tuple(vb_handle, ib_handle, IndexBufferHandle());
}

Box LineSetBuffersBuilder::ComputeAABB() {
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
