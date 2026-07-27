// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
    // Make sure geometry is on CPU
    auto pts = geometry.GetPointPositions();
    if (pts.IsCUDA()) {
        utility::LogWarning(
                "GPU resident line sets are not currently supported for "
                "visualization. Copying data to CPU.");
        geometry_ = geometry.To(core::Device("CPU:0"));
    }

    // Make sure data types are Float32 for points
    if (pts.GetDtype() != core::Float32) {
        utility::LogWarning(
                "Tensor point cloud points must have DType of Float32 not {}. "
                "Converting.",
                pts.GetDtype().ToString());
        geometry_.GetPointPositions() = pts.To(core::Float32);
    }
    // Colors should be Float32 but will often by UInt8
    if (geometry_.HasLineColors() &&
        geometry_.GetLineColors().GetDtype() != core::Float32) {
        auto colors = geometry_.GetLineColors();
        geometry_.GetLineColors() = colors.To(core::Float32);
        if (colors.GetDtype() == core::UInt8) {
            geometry_.GetLineColors() = geometry_.GetLineColors() / 255.0f;
        }
    }
    // Make sure line indices are Uint32
    if (geometry_.HasLineIndices() &&
        geometry_.GetLineIndices().GetDtype() != core::UInt32) {
        auto indices = geometry_.GetLineIndices();
        geometry_.GetLineIndices() = indices.To(core::UInt32);
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

void TLineSetBuffersBuilder::ConstructThinLines(uint32_t& n_vertices,
                                                float** vertex_data,
                                                uint32_t& n_indices,
                                                uint32_t& indices_bytes,
                                                uint32_t** line_indices) {
    const auto& points = geometry_.GetPointPositions();
    const uint32_t n_elements = 7;
    const uint32_t vertex_stride = n_elements * sizeof(float);
    float* vdata;
    uint32_t* idata;

    // Two separate paths for lines with colors and those without
    if (geometry_.HasLineColors()) {
        // NOTE: The following code naively duplicates vertex positions for each
        // line in case there are multiple different colored lines sharing a
        // vertex. This could be made more intelligent to avoid unnecessary
        // duplication but as a practical matter there shouldn't be much if any
        // performance difference. This can be revisited in the future if
        // necessary.
        const auto& lines = geometry_.GetLineIndices();
        const auto& colors = geometry_.GetLineColors();
        n_vertices = lines.GetLength() * 2;
        core::Tensor filament_data =
                core::Tensor::Empty({n_vertices, n_elements}, core::Float32);
        filament_data.Slice(1, 0, 3) =
                points.IndexGet({lines.Reshape({n_vertices}).To(core::Int64)});
        filament_data.Slice(0, 0, n_vertices, 2).Slice(1, 3, 6) = colors;
        filament_data.Slice(0, 1, n_vertices, 2).Slice(1, 3, 6) = colors;
        filament_data.Slice(1, 6, 7) = 1.f;
        vdata = static_cast<float*>(malloc(n_vertices * vertex_stride));
        memcpy(vdata, filament_data.GetDataPtr(), n_vertices * vertex_stride);
        indices_bytes = n_vertices * sizeof(IndexType);
        n_indices = n_vertices;
        idata = static_cast<IndexType*>(malloc(indices_bytes));
        std::iota(idata, idata + n_vertices, 0);
    } else {
        n_vertices = points.GetLength();
        core::Tensor filament_data =
                core::Tensor::Empty({n_vertices, n_elements}, core::Float32);
        filament_data.Slice(1, 0, 3) = points;
        filament_data.Slice(1, 3, 7) = 1.f;
        const auto vertex_array_size = n_vertices * vertex_stride;
        vdata = static_cast<float*>(malloc(vertex_array_size));
        memcpy(vdata, filament_data.GetDataPtr(), vertex_array_size);
        indices_bytes =
                geometry_.GetLineIndices().GetLength() * 2 * sizeof(IndexType);
        n_indices = geometry_.GetLineIndices().GetLength() * 2;
        idata = static_cast<IndexType*>(malloc(indices_bytes));
        memcpy(idata, geometry_.GetLineIndices().GetDataPtr(), indices_bytes);
    }

    // Assign buffers back to inputs
    *vertex_data = vdata;
    *line_indices = idata;
}

void TLineSetBuffersBuilder::ConstructWideLines(uint32_t& n_vertices,
                                                float** vertex_data,
                                                uint32_t& n_indices,
                                                uint32_t& indices_bytes,
                                                uint32_t** line_indices) {
    const auto& points = geometry_.GetPointPositions();
    const auto& lines = geometry_.GetLineIndices();

    const uint32_t n_elements = 11;
    const uint32_t vertex_stride = n_elements * sizeof(float);
    float* vdata;
    uint32_t* idata;

    n_vertices = lines.GetLength() * 4;
    core::Tensor filament_data =
            core::Tensor::Empty({n_vertices, n_elements}, core::Float32);
    // Get the start and end vertex of each line
    core::Tensor pos1_data =
            points.IndexGet({lines.Slice(1, 0, 1)
                                     .Reshape({lines.GetLength()})
                                     .To(core::Int64)});
    core::Tensor pos2_data =
            points.IndexGet({lines.Slice(1, 1, 2)
                                     .Reshape({lines.GetLength()})
                                     .To(core::Int64)});
    // Fill the vertices. The original vertices get expanded to 4 vertices
    // for the 4 corners of the line (composed of 2 triangles) as follows:
    // pos1 pos2 1.0 [color]
    // pos1 pos2 -1.0 [color]
    // pos2 pos1 -1.0 [color]
    // pos2 pos1 1.0 [color]
    // Vertex
    filament_data.Slice(0, 0, n_vertices, 4).Slice(1, 0, 3) = pos1_data;
    filament_data.Slice(0, 1, n_vertices, 4).Slice(1, 0, 3) = pos1_data;
    filament_data.Slice(0, 2, n_vertices, 4).Slice(1, 0, 3) = pos2_data;
    filament_data.Slice(0, 3, n_vertices, 4).Slice(1, 0, 3) = pos2_data;
    // Next parameter
    filament_data.Slice(0, 0, n_vertices, 4).Slice(1, 3, 6) = pos2_data;
    filament_data.Slice(0, 1, n_vertices, 4).Slice(1, 3, 6) = pos2_data;
    filament_data.Slice(0, 2, n_vertices, 4).Slice(1, 3, 6) = pos1_data;
    filament_data.Slice(0, 3, n_vertices, 4).Slice(1, 3, 6) = pos1_data;
    // Direction parameter
    filament_data.Slice(0, 0, n_vertices, 4).Slice(1, 6, 7) = 1.f;
    filament_data.Slice(0, 1, n_vertices, 4).Slice(1, 6, 7) = -1.f;
    filament_data.Slice(0, 2, n_vertices, 4).Slice(1, 6, 7) = -1.f;
    filament_data.Slice(0, 3, n_vertices, 4).Slice(1, 6, 7) = 1.f;
    // Fill in color
    if (geometry_.HasLineColors()) {
        const auto& colors = geometry_.GetLineColors();
        filament_data.Slice(0, 0, n_vertices, 4).Slice(1, 7, 10) = colors;
        filament_data.Slice(0, 1, n_vertices, 4).Slice(1, 7, 10) = colors;
        filament_data.Slice(0, 2, n_vertices, 4).Slice(1, 7, 10) = colors;
        filament_data.Slice(0, 3, n_vertices, 4).Slice(1, 7, 10) = colors;
        filament_data.Slice(1, 10, 11) = 1.f;  // alpha value
    } else {
        filament_data.Slice(1, 7, 11) = 1.f;
    }

    // Copy per-vertex data to output array
    const auto vertex_array_size = n_vertices * vertex_stride;
    vdata = static_cast<float*>(malloc(vertex_array_size));
    memcpy(vdata, filament_data.GetDataPtr(), vertex_array_size);

    // Build the triangles for the wide lines
    n_indices = geometry_.GetLineIndices().GetLength() * 6;
    indices_bytes = n_indices * sizeof(IndexType);
    idata = static_cast<IndexType*>(malloc(indices_bytes));
    for (uint32_t i = 0, vertex_idx = 0; i < n_indices; vertex_idx += 4) {
        // Triangle 1
        idata[i++] = vertex_idx;
        idata[i++] = vertex_idx + 1;
        idata[i++] = vertex_idx + 2;
        // Triangle 2
        idata[i++] = vertex_idx + 3;
        idata[i++] = vertex_idx + 2;
        idata[i++] = vertex_idx + 1;
    }

    // Assign buffers back to inputs
    *vertex_data = vdata;
    *line_indices = idata;
}

GeometryBuffersBuilder::Buffers TLineSetBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const uint32_t n_elements = wide_lines_ ? 11 : 7;
    const uint32_t vertex_stride = n_elements * sizeof(float);
    const uint32_t vertex_start_offset = 0;
    const uint32_t next_start_offest = 3 * sizeof(float);
    const uint32_t color_start_offset =
            wide_lines_ ? 7 * sizeof(float) : 3 * sizeof(float);

    uint32_t n_vertices = 0;
    float* vertex_data = nullptr;
    uint32_t n_indices = 0;
    IndexType* line_indices = nullptr;
    uint32_t indices_bytes = 0;

    // Separate paths for thin and wide lines
    if (wide_lines_) {
        ConstructWideLines(n_vertices, &vertex_data, n_indices, indices_bytes,
                           &line_indices);
    } else {
        ConstructThinLines(n_vertices, &vertex_data, n_indices, indices_bytes,
                           &line_indices);
    }

    VertexBuffer* vbuf;
    // Different GPU vertex layouts for
    if (wide_lines_) {
        vbuf = VertexBuffer::Builder()
                       .bufferCount(1)
                       .vertexCount(n_vertices)
                       .attribute(VertexAttribute::POSITION, 0,
                                  VertexBuffer::AttributeType::FLOAT3,
                                  vertex_start_offset, vertex_stride)
                       .attribute(VertexAttribute::CUSTOM0, 0,
                                  VertexBuffer::AttributeType::FLOAT4,
                                  next_start_offest, vertex_stride)
                       .attribute(VertexAttribute::COLOR, 0,
                                  VertexBuffer::AttributeType::FLOAT4,
                                  color_start_offset, vertex_stride)
                       .build(engine);
    } else {
        vbuf = VertexBuffer::Builder()
                       .bufferCount(1)
                       .vertexCount(n_vertices)
                       .attribute(VertexAttribute::POSITION, 0,
                                  VertexBuffer::AttributeType::FLOAT3,
                                  vertex_start_offset, vertex_stride)
                       .attribute(VertexAttribute::COLOR, 0,
                                  VertexBuffer::AttributeType::FLOAT4,
                                  color_start_offset, vertex_stride)
                       .build(engine);
    }

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    // Create vertex and index buffer
    VertexBuffer::BufferDescriptor vb_descriptor(vertex_data,
                                                 n_vertices * vertex_stride);
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    auto ib_handle =
            resource_mgr.CreateIndexBuffer(n_indices, sizeof(IndexType));
    if (!ib_handle) {
        free(line_indices);
        return {};
    }
    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();
    IndexBuffer::BufferDescriptor ib_descriptor(line_indices, indices_bytes);
    ib_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(ib_descriptor));

    return {vb_handle, ib_handle, IndexBufferHandle()};
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
