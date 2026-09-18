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

#include <algorithm>
#include <map>
#include <numeric>

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/TriangleMesh.h"
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
                    .attribute(VertexAttribute::TANGENTS, 0,
                               VertexBuffer::AttributeType::FLOAT4,
                               GetVertexTangentOffset<TexturedVertex>(), stride)
                    .attribute(VertexAttribute::CUSTOM0, 0,
                               VertexBuffer::AttributeType::FLOAT4,
                               GetVertexTangentOffset<TexturedVertex>(),
                               stride);

    if (has_colors) {
        builder.attribute(VertexAttribute::COLOR, 0,
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
    vertex_data.byte_count =
            vertex_data.vertices_count * sizeof(TexturedVertex);
    vertex_data.bytes_to_copy = vertex_data.byte_count;
    vertex_data.bytes = malloc(vertex_data.byte_count);

    const TexturedVertex kDefault;
    auto vertices = static_cast<TexturedVertex*>(vertex_data.bytes);
    for (size_t i = 0; i < vertex_data.vertices_count; ++i) {
        TexturedVertex& element = vertices[i];

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
std::tuple<vbdata, ibdata> CreateFromDuplicatedMesh(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertex_data;
    ibdata index_data;

    index_data.stride = sizeof(GeometryBuffersBuilder::IndexType);
    index_data.byte_count = geometry.triangles_.size() * 3 * index_data.stride;
    index_data.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(index_data.byte_count));
    GeometryBuffersBuilder::IndexType* index_ptr = index_data.bytes;

    vertex_data.byte_count =
            geometry.triangles_.size() * 3 * sizeof(TexturedVertex);
    vertex_data.bytes_to_copy = vertex_data.byte_count;
    vertex_data.bytes = malloc(vertex_data.byte_count);
    vertex_data.vertices_count = geometry.triangles_.size() * 3;

    const TexturedVertex kDefault;
    auto textured_vertices = static_cast<TexturedVertex*>(vertex_data.bytes);
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];

        for (size_t j = 0; j < 3; ++j) {
            GeometryBuffersBuilder::IndexType index = triangle(j);

            auto uv = geometry.triangle_uvs_[i * 3 + j];
            auto pos = geometry.vertices_[index];

            TexturedVertex& element = textured_vertices[index];
            SetVertexPosition(element, pos);
            if (tangents != nullptr) {
                element.tangent = tangents[index];
            } else {
                element.tangent = kDefault.tangent;
            }

            SetVertexUV(element, uv);

            if (geometry.HasVertexColors()) {
                SetVertexColor(element, geometry.vertex_colors_[index]);
            } else {
                element.color = kDefault.color;
            }
            *index_ptr++ = index;
        }
    }

    return std::make_tuple(vertex_data, index_data);
}

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

    const geometry::TriangleMesh* internal_geom = &geometry_;
    geometry::TriangleMesh duplicated_mesh;
    if (geometry_.HasTriangleNormals() && !geometry_.HasVertexNormals()) {
        // If the TriangleMesh has per-triangle normals then the normals must be
        // converted to per-vertex and vertices/Uvs duplicated.
        const size_t new_vertex_count = geometry_.triangles_.size() * 3;
        duplicated_mesh.vertices_.reserve(new_vertex_count);
        duplicated_mesh.vertex_normals_.reserve(new_vertex_count);
        if (geometry_.HasVertexColors()) {
            duplicated_mesh.vertex_colors_.reserve(new_vertex_count);
        }
        for (unsigned int i = 0; i < geometry_.triangles_.size(); ++i) {
            auto& tri_idx = geometry_.triangles_[i];
            duplicated_mesh.vertices_.push_back(
                    geometry_.vertices_[tri_idx.x()]);
            duplicated_mesh.vertices_.push_back(
                    geometry_.vertices_[tri_idx.y()]);
            duplicated_mesh.vertices_.push_back(
                    geometry_.vertices_[tri_idx.z()]);
            duplicated_mesh.vertex_normals_.push_back(
                    geometry_.triangle_normals_[i]);
            duplicated_mesh.vertex_normals_.push_back(
                    geometry_.triangle_normals_[i]);
            duplicated_mesh.vertex_normals_.push_back(
                    geometry_.triangle_normals_[i]);
            if (geometry_.HasVertexColors()) {
                duplicated_mesh.vertex_colors_.push_back(
                        geometry_.vertex_colors_[tri_idx.x()]);
                duplicated_mesh.vertex_colors_.push_back(
                        geometry_.vertex_colors_[tri_idx.y()]);
                duplicated_mesh.vertex_colors_.push_back(
                        geometry_.vertex_colors_[tri_idx.z()]);
            }
            duplicated_mesh.triangles_.push_back(
                    Eigen::Vector3i(i * 3, i * 3 + 1, i * 3 + 2));
        }
        // utility::LogWarning("Taking the duplicate mesh path!");
        duplicated_mesh.triangle_uvs_ = geometry_.triangle_uvs_;
        internal_geom = &duplicated_mesh;
    }

    const size_t n_vertices = internal_geom->vertices_.size();

    math::quatf* float4v_tangents = nullptr;
    if (internal_geom->HasVertexNormals()) {
        // Converting vertex normals to float base
        std::vector<Eigen::Vector3f> normals;
        normals.resize(n_vertices);
        for (size_t i = 0; i < n_vertices; ++i) {
            normals[i] = internal_geom->vertex_normals_[i].cast<float>();
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
        delete orientation;
    }

    // NOTE: Both default lit and unlit material shaders require per-vertex
    // colors so we unconditionally assume the triangle mesh has color.
    const bool has_colors = true;
    bool has_uvs = internal_geom->HasTriangleUvs();
    bool using_duplicated_mesh = duplicated_mesh.vertices_.size() > 0;

    // We take ownership of vbdata.bytes and ibdata.bytes here.
    std::tuple<vbdata, ibdata> buffers_data;
    size_t stride = sizeof(BaseVertex);
    if (has_uvs && using_duplicated_mesh) {
        buffers_data =
                CreateFromDuplicatedMesh(float4v_tangents, *internal_geom);
        stride = sizeof(TexturedVertex);
    } else if (has_uvs) {
        buffers_data = CreateTexturedBuffers(float4v_tangents, *internal_geom);
        stride = sizeof(TexturedVertex);
    } else if (has_colors) {
        buffers_data = CreateColoredBuffers(float4v_tangents, *internal_geom);
        stride = sizeof(TexturedVertex);
        has_uvs = true;
    } else {
        buffers_data = CreatePlainBuffers(float4v_tangents, *internal_geom);
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

TMeshBuffersBuilder::TMeshBuffersBuilder(
        const t::geometry::TriangleMesh& geometry)
    : geometry_(geometry) {
    if (!geometry.GetDevice().IsCPU()) {
        utility::LogWarning(
                "Non-CPU tensor triangle meshes are not currently supported "
                "for visualization. Copying data to CPU.");
        geometry_ = geometry.To(core::Device("CPU:0"));
    }

    auto& points = geometry_.GetVertexPositions();
    if (points.GetDtype() != core::Float32) {
        utility::LogWarning(
                "Tensor triangle mesh vertices must have DType of Float32 not "
                "{}. Converting.",
                points.GetDtype().ToString());
    }
    points = points.To(core::Float32).Contiguous();

    auto& indices = geometry_.GetTriangleIndices();
    indices = indices.To(core::UInt32).Contiguous();

    if (geometry_.HasVertexNormals()) {
        geometry_.GetVertexNormals() =
                geometry_.GetVertexNormals().To(core::Float32).Contiguous();
    }
    if (geometry_.HasTriangleNormals()) {
        geometry_.GetTriangleNormals() =
                geometry_.GetTriangleNormals().To(core::Float32).Contiguous();
    }

    auto normalize_colors = [](core::Tensor colors) {
        const bool is_uint8 = colors.GetDtype() == core::UInt8;
        colors = colors.To(core::Float32).Contiguous();
        return is_uint8 ? colors / 255.0f : colors;
    };
    if (geometry_.HasVertexColors()) {
        geometry_.GetVertexColors() =
                normalize_colors(geometry_.GetVertexColors());
    }
    if (geometry_.HasTriangleColors()) {
        geometry_.GetTriangleColors() =
                normalize_colors(geometry_.GetTriangleColors());
    }
    if (geometry_.HasVertexAttr("texture_uvs")) {
        geometry_.GetVertexAttr("texture_uvs") =
                geometry_.GetVertexAttr("texture_uvs")
                        .To(core::Float32)
                        .Contiguous();
    }
    if (geometry_.HasTriangleAttr("texture_uvs")) {
        geometry_.GetTriangleAttr("texture_uvs") =
                geometry_.GetTriangleAttr("texture_uvs")
                        .To(core::Float32)
                        .Contiguous();
    }
}

RenderableManager::PrimitiveType TMeshBuffersBuilder::GetPrimitiveType() const {
    return RenderableManager::PrimitiveType::TRIANGLES;
}

bool TMeshBuffersBuilder::RequiresVertexDuplication() const {
    return geometry_.HasTriangleNormals() || geometry_.HasTriangleColors() ||
           geometry_.HasTriangleAttr("texture_uvs");
}

size_t TMeshBuffersBuilder::GetVertexCount() const {
    return RequiresVertexDuplication()
                   ? geometry_.GetTriangleIndices().GetLength() * 3
                   : geometry_.GetVertexPositions().GetLength();
}

size_t TMeshBuffersBuilder::GetIndexCount() const {
    return geometry_.GetTriangleIndices().GetLength() * 3;
}

uint64_t TMeshBuffersBuilder::GetTopologyHash() const {
    const auto& indices = geometry_.GetTriangleIndices();
    const auto* data = indices.GetDataPtr<uint32_t>();
    const size_t count = GetIndexCount();
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

GeometryBuffersBuilder::Buffers TMeshBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();
    const size_t n_vertices = GetVertexCount();

    // We use CUSTOM0 for tangents along with TANGENTS attribute
    // because Filament would optimize out anything about normals and lightning
    // from unlit materials. But our shader for normals visualizing is unlit, so
    // we need to use this workaround.
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(4)
                                 .vertexCount(uint32_t(n_vertices))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .attribute(VertexAttribute::COLOR, 1,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .attribute(VertexAttribute::TANGENTS, 2,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM0, 2,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::UV0, 3,
                                            VertexBuffer::AttributeType::FLOAT2)
                                 .build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    // Create the index buffer
    // NOTE: Filament supports both UInt16 and UInt32 triangle indices.
    // Currently, however, we only support 32bit indices. This may change in the
    // future.
    auto ib_handle =
            resource_mgr.CreateIndexBuffer(GetIndexCount(), sizeof(uint32_t));
    if (!ib_handle ||
        !UpdateBuffers(vb_handle, ib_handle, true, true, true, true, true)) {
        if (ib_handle) resource_mgr.Destroy(ib_handle);
        resource_mgr.Destroy(vb_handle);
        return {};
    }
    IndexBufferHandle downsampled_handle;

    return std::make_tuple(vb_handle, ib_handle, downsampled_handle);
}

bool TMeshBuffersBuilder::UpdateBuffers(VertexBufferHandle vertex_buffer,
                                        IndexBufferHandle index_buffer,
                                        bool update_positions,
                                        bool update_normals,
                                        bool update_colors,
                                        bool update_uvs,
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
                "Tensor triangle mesh update requires unchanged render vertex "
                "and index counts (vertices: expected {}, available {}; "
                "indices: expected {}, available {}).",
                n_vertices, vbuf ? vbuf->getVertexCount() : 0, n_indices,
                ibuf ? ibuf->getIndexCount() : 0);
        return false;
    }

    const bool duplicate_vertices = RequiresVertexDuplication();
    const auto& points = geometry_.GetVertexPositions();
    const auto& indices = geometry_.GetTriangleIndices();
    const auto indices_64 = indices.To(core::Int64);
    const auto flat_indices =
            indices_64.Reshape({static_cast<int64_t>(n_indices)});

    if (update_positions) {
        const size_t array_size = n_vertices * 3 * sizeof(float);
        auto* data = static_cast<float*>(malloc(array_size));
        if (duplicate_vertices) {
            const auto expanded = points.IndexGet({flat_indices});
            memcpy(data, expanded.GetDataPtr(), array_size);
        } else {
            memcpy(data, points.GetDataPtr(), array_size);
        }
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 0, std::move(descriptor));
    }

    if (update_colors) {
        const size_t array_size = n_vertices * 3 * sizeof(float);
        auto* data = static_cast<float*>(malloc(array_size));
        if (geometry_.HasVertexColors()) {
            if (duplicate_vertices) {
                const auto expanded =
                        geometry_.GetVertexColors().IndexGet({flat_indices});
                memcpy(data, expanded.GetDataPtr(), array_size);
            } else {
                memcpy(data, geometry_.GetVertexColors().GetDataPtr(),
                       array_size);
            }
        } else if (geometry_.HasTriangleColors()) {
            const auto& colors = geometry_.GetTriangleColors();
            core::Tensor expanded = core::Tensor::Empty(
                    {static_cast<int64_t>(n_vertices), 3}, core::Float32);
            expanded.Slice(0, 0, n_vertices, 3) = colors;
            expanded.Slice(0, 1, n_vertices, 3) = colors;
            expanded.Slice(0, 2, n_vertices, 3) = colors;
            memcpy(data, expanded.GetDataPtr(), array_size);
        } else {
            std::fill(data, data + n_vertices * 3, 0.5f);
        }
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 1, std::move(descriptor));
    }

    if (update_normals) {
        const size_t array_size = n_vertices * 4 * sizeof(float);
        auto* data = static_cast<float*>(malloc(array_size));
        const float* normals = nullptr;
        core::Tensor expanded;
        if (geometry_.HasVertexNormals()) {
            if (duplicate_vertices) {
                expanded =
                        geometry_.GetVertexNormals().IndexGet({flat_indices});
                normals = expanded.GetDataPtr<float>();
            } else {
                normals = geometry_.GetVertexNormals().GetDataPtr<float>();
            }
        } else if (geometry_.HasTriangleNormals()) {
            const auto& triangle_normals = geometry_.GetTriangleNormals();
            expanded = core::Tensor::Empty(
                    {static_cast<int64_t>(n_vertices), 3}, core::Float32);
            expanded.Slice(0, 0, n_vertices, 3) = triangle_normals;
            expanded.Slice(0, 1, n_vertices, 3) = triangle_normals;
            expanded.Slice(0, 2, n_vertices, 3) = triangle_normals;
            normals = expanded.GetDataPtr<float>();
        }
        if (normals) {
            auto orientation =
                    filament::geometry::SurfaceOrientation::Builder()
                            .vertexCount(n_vertices)
                            .normals(reinterpret_cast<const math::float3*>(
                                    normals))
                            .build();
            orientation->getQuats(reinterpret_cast<math::quatf*>(data),
                                  n_vertices);
            delete orientation;
        } else {
            auto* normal = data;
            for (size_t i = 0; i < n_vertices; ++i) {
                *normal++ = 0.f;
                *normal++ = 0.f;
                *normal++ = 0.f;
                *normal++ = 1.f;
            }
        }
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 2, std::move(descriptor));
    }

    if (update_uvs) {
        const size_t array_size = n_vertices * 2 * sizeof(float);
        auto* data = static_cast<float*>(malloc(array_size));
        if (geometry_.HasVertexAttr("texture_uvs")) {
            if (duplicate_vertices) {
                const auto expanded = geometry_.GetVertexAttr("texture_uvs")
                                              .IndexGet({flat_indices});
                memcpy(data, expanded.GetDataPtr(), array_size);
            } else {
                memcpy(data,
                       geometry_.GetVertexAttr("texture_uvs").GetDataPtr(),
                       array_size);
            }
        } else if (geometry_.HasTriangleAttr("texture_uvs")) {
            memcpy(data, geometry_.GetTriangleAttr("texture_uvs").GetDataPtr(),
                   array_size);
        } else {
            memset(data, 0, array_size);
        }
        VertexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 3, std::move(descriptor));
    }

    if (update_indices) {
        const size_t array_size = n_indices * sizeof(uint32_t);
        auto* data = static_cast<uint32_t*>(malloc(array_size));
        if (duplicate_vertices) {
            std::iota(data, data + n_indices, 0);
        } else {
            const auto indices_32 = indices.To(core::UInt32);
            memcpy(data, indices_32.GetDataPtr(), array_size);
        }
        IndexBuffer::BufferDescriptor descriptor(
                data, array_size, GeometryBuffersBuilder::DeallocateBuffer);
        ibuf->setBuffer(engine, std::move(descriptor));
    }
    return true;
}

filament::Box TMeshBuffersBuilder::ComputeAABB() {
    auto min_bounds = geometry_.GetMinBound();
    auto max_bounds = geometry_.GetMaxBound();
    auto* min_bounds_float = min_bounds.GetDataPtr<float>();
    auto* max_bounds_float = max_bounds.GetDataPtr<float>();

    const filament::math::float3 min_pt(
            min_bounds_float[0], min_bounds_float[1], min_bounds_float[2]);
    const filament::math::float3 max_pt(
            max_bounds_float[0], max_bounds_float[1], max_bounds_float[2]);

    Box aabb;
    aabb.set(min_pt, max_pt);
    if (aabb.isEmpty()) {
        const filament::math::float3 offset(0.1, 0.1, 0.1);
        aabb.set(min_pt - offset, max_pt + offset);
    }

    return aabb;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
