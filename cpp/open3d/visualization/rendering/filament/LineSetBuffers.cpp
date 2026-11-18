// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
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

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>

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

TLineSetBuffersBuilder::TLineSetBuffersBuilder(
        const t::geometry::LineSet& geometry)
    : geometry_(geometry) {
    if (!geometry.GetDevice().IsCPU()) {
        utility::LogWarning(
                "Non-CPU tensor line sets are not currently supported for "
                "visualization. Copying data to CPU.");
        geometry_ = geometry.To(core::Device("CPU:0"));
    }

    auto& pts = geometry_.GetPointPositions();
    if (pts.GetDtype() != core::Float32) {
        utility::LogWarning(
                "Tensor line set points must have DType of Float32 not {}. "
                "Converting.",
                pts.GetDtype().ToString());
    }
    pts = pts.To(core::Float32).Contiguous();

    auto& indices = geometry_.GetLineIndices();
    indices = indices.To(core::UInt32).Contiguous();

    if (geometry_.HasLineColors()) {
        auto& colors = geometry_.GetLineColors();
        const bool is_uint8 = colors.GetDtype() == core::UInt8;
        colors = colors.To(core::Float32).Contiguous();
        if (is_uint8) colors = colors / 255.0f;
    }
}

RenderableManager::PrimitiveType TLineSetBuffersBuilder::GetPrimitiveType()
        const {
    if (wide_lines_) {
        return RenderableManager::PrimitiveType::TRIANGLES;
    } else {
        return RenderableManager::PrimitiveType::LINES;
    }
}

TLineSetBuffersBuilder::Layout TLineSetBuffersBuilder::GetLayout() const {
    if (wide_lines_) return Layout::kWideExpanded;
    return geometry_.HasLineColors() ? Layout::kThinExpanded
                                     : Layout::kThinIndexed;
}

size_t TLineSetBuffersBuilder::GetVertexCount() const {
    switch (GetLayout()) {
        case Layout::kThinIndexed:
            return geometry_.GetPointPositions().GetLength();
        case Layout::kThinExpanded:
            return geometry_.GetLineIndices().GetLength() * 2;
        case Layout::kWideExpanded:
            return geometry_.GetLineIndices().GetLength() * 4;
    }
    return 0;
}

size_t TLineSetBuffersBuilder::GetIndexCount() const {
    return geometry_.GetLineIndices().GetLength() *
           (GetLayout() == Layout::kWideExpanded ? 6 : 2);
}

uint64_t TLineSetBuffersBuilder::GetTopologyHash() const {
    const auto& indices = geometry_.GetLineIndices();
    const auto* data = indices.GetDataPtr<uint32_t>();
    const size_t count = indices.GetLength() * 2;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

GeometryBuffersBuilder::Buffers TLineSetBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();
    const size_t n_vertices = GetVertexCount();
    const size_t n_indices = GetIndexCount();
    if (n_vertices > std::numeric_limits<uint32_t>::max() ||
        n_indices > std::numeric_limits<uint32_t>::max()) {
        utility::LogWarning(
                "Tensor line set is too large for Filament's 32-bit buffer "
                "counts.");
        return {};
    }

    VertexBuffer::Builder builder;
    builder.bufferCount(wide_lines_ ? 3 : 2)
            .vertexCount(static_cast<uint32_t>(n_vertices))
            .attribute(VertexAttribute::POSITION, 0,
                       VertexBuffer::AttributeType::FLOAT3)
            .attribute(VertexAttribute::COLOR, wide_lines_ ? 2 : 1,
                       VertexBuffer::AttributeType::FLOAT4);
    if (wide_lines_) {
        builder.attribute(VertexAttribute::CUSTOM0, 1,
                          VertexBuffer::AttributeType::FLOAT4);
    }
    VertexBuffer* vbuf = builder.build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    auto ib_handle =
            resource_mgr.CreateIndexBuffer(n_indices, sizeof(IndexType));
    if (!ib_handle || !UpdateBuffers(vb_handle, ib_handle, true, true, true)) {
        if (ib_handle) resource_mgr.Destroy(ib_handle);
        resource_mgr.Destroy(vb_handle);
        return {};
    }

    return {vb_handle, ib_handle, IndexBufferHandle()};
}

bool TLineSetBuffersBuilder::UpdateBuffers(VertexBufferHandle vertex_buffer,
                                           IndexBufferHandle index_buffer,
                                           bool update_points,
                                           bool update_colors,
                                           bool update_indices) {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();
    auto vbuf = resource_mgr.GetVertexBuffer(vertex_buffer).lock();
    auto ibuf = resource_mgr.GetIndexBuffer(index_buffer).lock();
    const size_t n_vertices = GetVertexCount();
    const size_t n_indices = GetIndexCount();
    if (!vbuf || !ibuf || vbuf->getVertexCount() != n_vertices ||
        ibuf->getIndexCount() != n_indices) {
        utility::LogWarning(
                "Tensor line set update requires unchanged render vertex and "
                "index counts (vertices: expected {}, available {}; indices: "
                "expected {}, available {}).",
                n_vertices, vbuf ? vbuf->getVertexCount() : 0, n_indices,
                ibuf ? ibuf->getIndexCount() : 0);
        return false;
    }

    const auto layout = GetLayout();
    const auto& points = geometry_.GetPointPositions();
    const auto& lines = geometry_.GetLineIndices();
    const auto flat_lines =
            lines.To(core::Int64)
                    .Reshape({static_cast<int64_t>(
                            geometry_.GetLineIndices().GetLength() * 2)});

    core::Tensor pos1;
    core::Tensor pos2;
    if (layout == Layout::kWideExpanded) {
        const auto n_lines = static_cast<int64_t>(lines.GetLength());
        pos1 = points.IndexGet(
                {lines.Slice(1, 0, 1).Reshape({n_lines}).To(core::Int64)});
        pos2 = points.IndexGet(
                {lines.Slice(1, 1, 2).Reshape({n_lines}).To(core::Int64)});
    }

    if (update_points) {
        core::Tensor packed;
        if (layout == Layout::kThinIndexed) {
            packed = points;
        } else if (layout == Layout::kThinExpanded) {
            packed = points.IndexGet({flat_lines});
        } else {
            packed = core::Tensor::Empty({static_cast<int64_t>(n_vertices), 3},
                                         core::Float32);
            packed.Slice(0, 0, n_vertices, 4) = pos1;
            packed.Slice(0, 1, n_vertices, 4) = pos1;
            packed.Slice(0, 2, n_vertices, 4) = pos2;
            packed.Slice(0, 3, n_vertices, 4) = pos2;
        }
        const size_t array_size = n_vertices * 3 * sizeof(float);
        auto* data = malloc(array_size);
        memcpy(data, packed.GetDataPtr(), array_size);
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 0, std::move(descriptor));

        if (layout == Layout::kWideExpanded) {
            core::Tensor next = core::Tensor::Empty(
                    {static_cast<int64_t>(n_vertices), 4}, core::Float32);
            next.Slice(0, 0, n_vertices, 4).Slice(1, 0, 3) = pos2;
            next.Slice(0, 1, n_vertices, 4).Slice(1, 0, 3) = pos2;
            next.Slice(0, 2, n_vertices, 4).Slice(1, 0, 3) = pos1;
            next.Slice(0, 3, n_vertices, 4).Slice(1, 0, 3) = pos1;
            next.Slice(0, 0, n_vertices, 4).Slice(1, 3, 4) = 1.f;
            next.Slice(0, 1, n_vertices, 4).Slice(1, 3, 4) = -1.f;
            next.Slice(0, 2, n_vertices, 4).Slice(1, 3, 4) = -1.f;
            next.Slice(0, 3, n_vertices, 4).Slice(1, 3, 4) = 1.f;
            const size_t next_size = n_vertices * 4 * sizeof(float);
            auto* next_data = malloc(next_size);
            memcpy(next_data, next.GetDataPtr(), next_size);
            VertexBuffer::BufferDescriptor next_descriptor(
                    next_data, next_size,
                    GeometryBuffersBuilder::DeallocateBuffer);
            vbuf->setBufferAt(engine, 1, std::move(next_descriptor));
        }
    }

    if (update_colors) {
        core::Tensor packed = core::Tensor::Ones(
                {static_cast<int64_t>(n_vertices), 4}, core::Float32);
        if (geometry_.HasLineColors()) {
            const auto& colors = geometry_.GetLineColors();
            const int64_t stride =
                    layout == Layout::kWideExpanded ? int64_t{4} : int64_t{2};
            for (int64_t offset = 0; offset < stride; ++offset) {
                packed.Slice(0, offset, n_vertices, stride).Slice(1, 0, 3) =
                        colors;
            }
        }
        const size_t array_size = n_vertices * 4 * sizeof(float);
        auto* data = malloc(array_size);
        memcpy(data, packed.GetDataPtr(), array_size);
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, wide_lines_ ? 2 : 1, std::move(descriptor));
    }

    if (update_indices) {
        const size_t array_size = n_indices * sizeof(IndexType);
        auto* data = static_cast<IndexType*>(malloc(array_size));
        if (layout == Layout::kThinIndexed) {
            memcpy(data, lines.GetDataPtr(), array_size);
        } else if (layout == Layout::kThinExpanded) {
            std::iota(data, data + n_indices, 0);
        } else {
            for (uint32_t i = 0, vertex_idx = 0; i < n_indices;
                 vertex_idx += 4) {
                data[i++] = vertex_idx;
                data[i++] = vertex_idx + 1;
                data[i++] = vertex_idx + 2;
                data[i++] = vertex_idx + 3;
                data[i++] = vertex_idx + 2;
                data[i++] = vertex_idx + 1;
            }
        }
        IndexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        ibuf->setBuffer(engine, std::move(descriptor));
    }
    return true;
}

filament::Box TLineSetBuffersBuilder::ComputeAABB() {
    auto min_bounds = geometry_.GetMinBound();
    auto max_bounds = geometry_.GetMaxBound();
    auto* min_bounds_float = min_bounds.GetDataPtr<float>();
    auto* max_bounds_float = max_bounds.GetDataPtr<float>();

    const filament::math::float3 min(min_bounds_float[0], min_bounds_float[1],
                                     min_bounds_float[2]);
    const filament::math::float3 max(max_bounds_float[0], max_bounds_float[1],
                                     max_bounds_float[2]);

    Box aabb;
    aabb.set(min, max);
    return aabb;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
