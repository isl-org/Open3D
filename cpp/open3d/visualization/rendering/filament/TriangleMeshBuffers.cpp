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

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/MaterialEnums.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <map>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

using namespace filament;

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

struct BaseVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 1.f};
};

struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 1.f};
    math::float4 color = {0.5f, 0.5f, 0.5f, 1.f};
};

struct TexturedVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 1.f};
    math::float4 color = {0.5f, 0.5f, 0.5f, 1.f};
    math::float2 uv = {0.f, 0.f};
};

template <typename VertexType>
void SetVertexPosition(VertexType& vertex, const Eigen::Vector3d& pos) {
    auto float_pos = pos.cast<float>();
    vertex.position.x = float_pos(0);
    vertex.position.y = float_pos(1);
    vertex.position.z = float_pos(2);
}

template <typename VertexType>
void SetVertexColor(VertexType& vertex, const Eigen::Vector3d& c) {
    auto float_color = c.cast<float>();
    vertex.color.x = float_color(0);
    vertex.color.y = float_color(1);
    vertex.color.z = float_color(2);
}

template <typename VertexType>
void SetVertexUV(VertexType& vertex, const Eigen::Vector2d& UV) {
    auto float_uv = UV.cast<float>();
    vertex.uv.x = float_uv(0);
    vertex.uv.y = float_uv(1);
}

template <typename VertexType>
std::uint32_t GetVertexPositionOffset() {
    return offsetof(VertexType, position);
}

template <typename VertexType>
std::uint32_t GetVertexTangentOffset() {
    return offsetof(VertexType, tangent);
}

template <typename VertexType>
std::uint32_t GetVertexColorOffset() {
    return offsetof(VertexType, color);
}

template <typename VertexType>
std::uint32_t GetVertexUVOffset() {
    return offsetof(VertexType, uv);
}

template <typename VertexType>
std::uint32_t GetVertexStride() {
    return sizeof(VertexType);
}

VertexBuffer* BuildFilamentVertexBuffer(filament::Engine& engine,
                                        const std::uint32_t vertices_count,
                                        const std::uint32_t stride,
                                        bool has_uvs,
                                        bool has_colors) {
    // For CUSTOM0 explanation, see FilamentGeometryBuffersBuilder.cpp
    // Note, that TANGENTS and CUSTOM0 is pointing on same data in buffer
    auto builder =
            VertexBuffer::Builder()
                    .bufferCount(1)
                    .vertexCount(vertices_count)
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT3,
                               GetVertexPositionOffset<TexturedVertex>(),
                               stride)
                    .normalized(VertexAttribute::TANGENTS)
                    .attribute(VertexAttribute::TANGENTS, 0,
                               VertexBuffer::AttributeType::FLOAT4,
                               GetVertexTangentOffset<TexturedVertex>(), stride)
                    .attribute(VertexAttribute::CUSTOM0, 0,
                               VertexBuffer::AttributeType::FLOAT4,
                               GetVertexTangentOffset<TexturedVertex>(),
                               stride);

    if (has_colors) {
        builder.normalized(VertexAttribute::COLOR)
                .attribute(VertexAttribute::COLOR, 0,
                           VertexBuffer::AttributeType::FLOAT4,
                           GetVertexColorOffset<TexturedVertex>(), stride);
    }

    if (has_uvs) {
        builder.attribute(VertexAttribute::UV0, 0,
                          VertexBuffer::AttributeType::FLOAT2,
                          GetVertexUVOffset<TexturedVertex>(), stride);
    }

    return builder.build(engine);
}

struct vbdata {
    size_t byte_count = 0;
    size_t bytes_to_copy = 0;
    void* bytes = nullptr;
    size_t vertices_count = 0;
};

struct ibdata {
    size_t byte_count = 0;
    GeometryBuffersBuilder::IndexType* bytes = nullptr;
    size_t stride = 0;
};

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreatePlainBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertex_data;
    ibdata index_data;

    vertex_data.vertices_count = geometry.vertices_.size();
    vertex_data.byte_count = vertex_data.vertices_count * sizeof(BaseVertex);
    vertex_data.bytes_to_copy = vertex_data.byte_count;
    vertex_data.bytes = malloc(vertex_data.byte_count);

    const BaseVertex kDefault;
    auto plain_vertices = static_cast<BaseVertex*>(vertex_data.bytes);
    for (size_t i = 0; i < vertex_data.vertices_count; ++i) {
        BaseVertex& element = plain_vertices[i];

        SetVertexPosition(element, geometry.vertices_[i]);
        if (tangents != nullptr) {
            element.tangent = tangents[i];
        } else {
            element.tangent = kDefault.tangent;
        }
    }

    index_data.stride = sizeof(GeometryBuffersBuilder::IndexType);
    index_data.byte_count = geometry.triangles_.size() * 3 * index_data.stride;
    index_data.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(index_data.byte_count));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        index_data.bytes[3 * i] = triangle(0);
        index_data.bytes[3 * i + 1] = triangle(1);
        index_data.bytes[3 * i + 2] = triangle(2);
    }

    return std::make_tuple(vertex_data, index_data);
}

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreateColoredBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertex_data;
    ibdata index_data;

    vertex_data.vertices_count = geometry.vertices_.size();
    vertex_data.byte_count = vertex_data.vertices_count * sizeof(ColoredVertex);
    vertex_data.bytes_to_copy = vertex_data.byte_count;
    vertex_data.bytes = malloc(vertex_data.byte_count);

    const ColoredVertex kDefault;
    auto colored_vertices = static_cast<ColoredVertex*>(vertex_data.bytes);
    for (size_t i = 0; i < vertex_data.vertices_count; ++i) {
        ColoredVertex& element = colored_vertices[i];

        SetVertexPosition(element, geometry.vertices_[i]);
        if (tangents != nullptr) {
            element.tangent = tangents[i];
        } else {
            element.tangent = kDefault.tangent;
        }

        if (geometry.HasVertexColors()) {
            SetVertexColor(element, geometry.vertex_colors_[i]);
        } else {
            element.color = kDefault.color;
        }
    }

    index_data.stride = sizeof(GeometryBuffersBuilder::IndexType);
    index_data.byte_count = geometry.triangles_.size() * 3 * index_data.stride;
    index_data.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(index_data.byte_count));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        index_data.bytes[3 * i] = triangle(0);
        index_data.bytes[3 * i + 1] = triangle(1);
        index_data.bytes[3 * i + 2] = triangle(2);
    }

    return std::make_tuple(vertex_data, index_data);
}

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreateTexturedBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertex_data;
    ibdata index_data;

    struct LookupKey {
        LookupKey() = default;
        explicit LookupKey(const Eigen::Vector3d& pos,
                           const Eigen::Vector2d& uv) {
            values[0] = pos.x();
            values[1] = pos.y();
            values[2] = pos.z();
            values[3] = uv.x();
            values[4] = uv.y();
        }

        // Not necessarily transitive for points within kEpsilon.
        // TODO: does this break sort and map?
        bool operator<(const LookupKey& other) const {
            for (int i = 0; i < 5; ++i) {
                double diff = abs(values[i] - other.values[i]);
                if (diff > kEpsilon) {
                    return values[i] < other.values[i];
                }
            }

            return false;
        }

        const double kEpsilon = 0.00001;
        double values[5] = {0};
    };
    //                           < real index  , source index >
    std::map<LookupKey, std::pair<GeometryBuffersBuilder::IndexType,
                                  GeometryBuffersBuilder::IndexType>>
            index_lookup;

    index_data.stride = sizeof(GeometryBuffersBuilder::IndexType);
    index_data.byte_count = geometry.triangles_.size() * 3 * index_data.stride;
    index_data.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(index_data.byte_count));

    vertex_data.byte_count =
            geometry.triangles_.size() * 3 * sizeof(TexturedVertex);
    vertex_data.bytes = malloc(vertex_data.byte_count);

    GeometryBuffersBuilder::IndexType free_idx = 0;
    GeometryBuffersBuilder::IndexType uv_idx = 0;
    auto textured_vertices = static_cast<TexturedVertex*>(vertex_data.bytes);

    const TexturedVertex kDefault;
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];

        for (size_t j = 0; j < 3; ++j) {
            GeometryBuffersBuilder::IndexType index = triangle(j);

            auto uv = geometry.triangle_uvs_[uv_idx];
            auto pos = geometry.vertices_[index];

            LookupKey lookup_key(pos, uv);
            auto found = index_lookup.find(lookup_key);
            if (found != index_lookup.end()) {
                index = found->second.first;
            } else {
                index = free_idx;
                GeometryBuffersBuilder::IndexType source_idx = triangle(j);

                index_lookup[lookup_key] = {free_idx, source_idx};
                ++free_idx;

                TexturedVertex& element = textured_vertices[index];
                SetVertexPosition(element, pos);
                if (tangents != nullptr) {
                    element.tangent = tangents[source_idx];
                } else {
                    element.tangent = kDefault.tangent;
                }

                SetVertexUV(element, uv);

                if (geometry.HasVertexColors()) {
                    SetVertexColor(element,
                                   geometry.vertex_colors_[source_idx]);
                } else {
                    element.color = kDefault.color;
                }
            }

            index_data.bytes[3 * i + j] = index;

            ++uv_idx;
        }
    }

    vertex_data.vertices_count = free_idx;
    vertex_data.bytes_to_copy =
            vertex_data.vertices_count * sizeof(TexturedVertex);

    return std::make_tuple(vertex_data, index_data);
}

}  // namespace

TriangleMeshBuffersBuilder::TriangleMeshBuffersBuilder(
        const geometry::TriangleMesh& geometry)
    : geometry_(geometry) {}

RenderableManager::PrimitiveType TriangleMeshBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::TRIANGLES;
}

GeometryBuffersBuilder::Buffers TriangleMeshBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const size_t n_vertices = geometry_.vertices_.size();

    math::quatf* float4v_tangents = nullptr;
    if (geometry_.HasVertexNormals()) {
        // Converting vertex normals to float base
        std::vector<Eigen::Vector3f> normals;
        normals.resize(n_vertices);
        for (size_t i = 0; i < n_vertices; ++i) {
            normals[i] = geometry_.vertex_normals_[i].cast<float>();
        }

        // Converting normals to Filament type - quaternions
        const size_t tangents_byte_count = n_vertices * 4 * sizeof(float);
        float4v_tangents =
                static_cast<math::quatf*>(malloc(tangents_byte_count));
        auto orientation = filament::geometry::SurfaceOrientation::Builder()
                                   .vertexCount(n_vertices)
                                   .normals(reinterpret_cast<math::float3*>(
                                           normals.data()))
                                   .build();
        orientation->getQuats(float4v_tangents, n_vertices);
    } else {
        utility::LogWarning(
                "Trying to create mesh without vertex normals. Shading would "
                "not work correctly. Consider to generate vertex normals "
                "first.");
    }

    // NOTE: Both default lit and unlit material shaders require per-vertex
    // colors so we unconditionally assume the triangle mesh has color.
    const bool has_colors = true;
    const bool has_uvs = geometry_.HasTriangleUvs();

    // We take ownership of vbdata.bytes and ibdata.bytes here.
    std::tuple<vbdata, ibdata> buffers_data;
    size_t stride = sizeof(BaseVertex);
    if (has_uvs) {
        buffers_data = CreateTexturedBuffers(float4v_tangents, geometry_);
        stride = sizeof(TexturedVertex);
    } else if (has_colors) {
        buffers_data = CreateColoredBuffers(float4v_tangents, geometry_);
        stride = sizeof(ColoredVertex);
    } else {
        buffers_data = CreatePlainBuffers(float4v_tangents, geometry_);
    }

    free(float4v_tangents);

    const vbdata& vertex_data = std::get<0>(buffers_data);
    const ibdata& index_data = std::get<1>(buffers_data);

    VertexBuffer* vbuf = nullptr;
    vbuf = BuildFilamentVertexBuffer(
            engine, std::uint32_t(vertex_data.vertices_count),
            std::uint32_t(stride), has_uvs, has_colors);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        free(vertex_data.bytes);
        free(index_data.bytes);

        return {};
    }

    // Gives ownership of vertexData.bytes to VertexBuffer, which will
    // be deallocated later with DeallocateBuffer.
    VertexBuffer::BufferDescriptor vb_descriptor(vertex_data.bytes,
                                                 vertex_data.bytes_to_copy);
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    auto ib_handle = resource_mgr.CreateIndexBuffer(
            index_data.byte_count / index_data.stride, index_data.stride);
    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();

    // Gives ownership of indexData.bytes to IndexBuffer, which will
    // be deallocated later with DeallocateBuffer.
    IndexBuffer::BufferDescriptor ib_descriptor(index_data.bytes,
                                                index_data.byte_count);
    ib_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(ib_descriptor));

    return std::make_tuple(vb_handle, ib_handle, IndexBufferHandle());
}

filament::Box TriangleMeshBuffersBuilder::ComputeAABB() {
    auto geometry_aabb = geometry_.GetAxisAlignedBoundingBox();

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
