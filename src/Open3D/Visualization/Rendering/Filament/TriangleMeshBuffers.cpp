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

#include "FilamentGeometryBuffersBuilder.h"

#include "FilamentEngine.h"
#include "FilamentResourceManager.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/TriangleMesh.h"

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/filament/MaterialEnums.h>
#include <filament/geometry/SurfaceOrientation.h>

#include <map>

using namespace filament;

namespace open3d {
namespace visualization {

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
    auto floatPos = pos.cast<float>();
    vertex.position.x = floatPos(0);
    vertex.position.y = floatPos(1);
    vertex.position.z = floatPos(2);
}

template <typename VertexType>
void SetVertexColor(VertexType& vertex, const Eigen::Vector3d& c) {
    auto floatColor = c.cast<float>();
    vertex.color.x = floatColor(0);
    vertex.color.y = floatColor(1);
    vertex.color.z = floatColor(2);
}

template <typename VertexType>
void SetVertexUV(VertexType& vertex, const Eigen::Vector2d& UV) {
    auto floatUV = UV.cast<float>();
    vertex.uv.x = floatUV(0);
    vertex.uv.y = floatUV(1);
}

template <typename VertexType>
size_t GetVertexPositionOffset() {
    return offsetof(VertexType, position);
}

template <typename VertexType>
size_t GetVertexTangentOffset() {
    return offsetof(VertexType, tangent);
}

template <typename VertexType>
size_t GetVertexColorOffset() {
    return offsetof(VertexType, color);
}

template <typename VertexType>
size_t GetVertexUVOffset() {
    return offsetof(VertexType, uv);
}

template <typename VertexType>
size_t GetVertexStride() {
    return sizeof(VertexType);
}

VertexBuffer* BuildFilamentVertexBuffer(filament::Engine& engine,
                                        const std::uint32_t verticesCount,
                                        const std::uint32_t stride,
                                        bool hasUvs,
                                        bool hasColors) {
    // For CUSTOM0 explanation, see FilamentGeometryBuffersBuilder.cpp
    // Note, that TANGENTS and CUSTOM0 is pointing on same data in buffer
    auto builder =
            VertexBuffer::Builder()
                    .bufferCount(1)
                    .vertexCount(verticesCount)
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

    if (hasColors) {
        builder.normalized(VertexAttribute::COLOR)
                .attribute(VertexAttribute::COLOR, 0,
                           VertexBuffer::AttributeType::FLOAT4,
                           GetVertexColorOffset<TexturedVertex>(), stride);
    }

    if (hasUvs) {
        builder.attribute(VertexAttribute::UV0, 0,
                          VertexBuffer::AttributeType::FLOAT2,
                          GetVertexUVOffset<TexturedVertex>(), stride);
    }

    return builder.build(engine);
}

struct vbdata {
    size_t bytesCount = 0;
    size_t bytesToCopy = 0;
    void* bytes = nullptr;
    size_t verticesCount = 0;
};

struct ibdata {
    size_t bytesCount = 0;
    GeometryBuffersBuilder::IndexType* bytes = nullptr;
    size_t stride = 0;
};

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreatePlainBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

    vertexData.verticesCount = geometry.vertices_.size();
    vertexData.bytesCount = vertexData.verticesCount * sizeof(BaseVertex);
    vertexData.bytesToCopy = vertexData.bytesCount;
    vertexData.bytes = malloc(vertexData.bytesCount);

    const BaseVertex kDefault;
    auto plainVertices = static_cast<BaseVertex*>(vertexData.bytes);
    for (size_t i = 0; i < vertexData.verticesCount; ++i) {
        BaseVertex& element = plainVertices[i];

        SetVertexPosition(element, geometry.vertices_[i]);
        if (tangents != nullptr) {
            element.tangent = tangents[i];
        } else {
            element.tangent = kDefault.tangent;
        }
    }

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(indexData.bytesCount));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        indexData.bytes[3 * i] = triangle(0);
        indexData.bytes[3 * i + 1] = triangle(1);
        indexData.bytes[3 * i + 2] = triangle(2);
    }

    return std::make_tuple(vertexData, indexData);
}

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreateColoredBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

    vertexData.verticesCount = geometry.vertices_.size();
    vertexData.bytesCount = vertexData.verticesCount * sizeof(ColoredVertex);
    vertexData.bytesToCopy = vertexData.bytesCount;
    vertexData.bytes = malloc(vertexData.bytesCount);

    const ColoredVertex kDefault;
    auto coloredVertices = static_cast<ColoredVertex*>(vertexData.bytes);
    for (size_t i = 0; i < vertexData.verticesCount; ++i) {
        ColoredVertex& element = coloredVertices[i];

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

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(indexData.bytesCount));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        indexData.bytes[3 * i] = triangle(0);
        indexData.bytes[3 * i + 1] = triangle(1);
        indexData.bytes[3 * i + 2] = triangle(2);
    }

    return std::make_tuple(vertexData, indexData);
}

// Transfers ownership on return for vbdata.bytes and ibdata.bytes
std::tuple<vbdata, ibdata> CreateTexturedBuffers(
        const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

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
            indexLookup;

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(
            malloc(indexData.bytesCount));

    vertexData.bytesCount =
            geometry.triangles_.size() * 3 * sizeof(TexturedVertex);
    vertexData.bytes = malloc(vertexData.bytesCount);

    GeometryBuffersBuilder::IndexType freeIndex = 0;
    GeometryBuffersBuilder::IndexType uvIndex = 0;
    auto texturedVertices = static_cast<TexturedVertex*>(vertexData.bytes);

    const TexturedVertex kDefault;
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];

        for (size_t j = 0; j < 3; ++j) {
            GeometryBuffersBuilder::IndexType index = triangle(j);

            auto uv = geometry.triangle_uvs_[uvIndex];
            auto pos = geometry.vertices_[index];

            LookupKey lookupKey(pos, uv);
            auto found = indexLookup.find(lookupKey);
            if (found != indexLookup.end()) {
                index = found->second.first;
            } else {
                index = freeIndex;
                GeometryBuffersBuilder::IndexType sourceIndex = triangle(j);

                indexLookup[lookupKey] = {freeIndex, sourceIndex};
                ++freeIndex;

                TexturedVertex& element = texturedVertices[index];
                SetVertexPosition(element, pos);
                if (tangents != nullptr) {
                    element.tangent = tangents[sourceIndex];
                } else {
                    element.tangent = kDefault.tangent;
                }

                SetVertexUV(element, uv);

                if (geometry.HasVertexColors()) {
                    SetVertexColor(element,
                                   geometry.vertex_colors_[sourceIndex]);
                } else {
                    element.color = kDefault.color;
                }
            }

            indexData.bytes[3 * i + j] = index;

            ++uvIndex;
        }
    }

    vertexData.verticesCount = freeIndex;
    vertexData.bytesToCopy = vertexData.verticesCount * sizeof(TexturedVertex);

    return std::make_tuple(vertexData, indexData);
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
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.vertices_.size();

    math::quatf* float4VTangents = nullptr;
    if (geometry_.HasVertexNormals()) {
        // Converting vertex normals to float base
        std::vector<Eigen::Vector3f> normals;
        normals.resize(nVertices);
        for (size_t i = 0; i < nVertices; ++i) {
            normals[i] = geometry_.vertex_normals_[i].cast<float>();
        }

        // Converting normals to Filament type - quaternions
        const size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
        float4VTangents = static_cast<math::quatf*>(malloc(tangentsBytesCount));
        auto orientation = filament::geometry::SurfaceOrientation::Builder()
                                   .vertexCount(nVertices)
                                   .normals(reinterpret_cast<math::float3*>(
                                           normals.data()))
                                   .build();
        orientation.getQuats(float4VTangents, nVertices);
    } else {
        utility::LogWarning(
                "Trying to create mesh without vertex normals. Shading would "
                "not work correctly. Consider to generate vertex normals "
                "first.");
    }

    // We default to using vertex color attribute for all geometries, even if
    // a geometry doesn't have one. That's all due to our default material and
    // large variety of geometries it should support
    const bool hasColors = true;  // geometry_.HasVertexColors();
    const bool hasUVs = geometry_.HasTriangleUvs();

    // We take ownership of vbdata.bytes and ibdata.bytes here.
    std::tuple<vbdata, ibdata> buffersData;
    if (hasUVs) {
        buffersData = CreateTexturedBuffers(float4VTangents, geometry_);
    } else if (hasColors) {
        buffersData = CreateColoredBuffers(float4VTangents, geometry_);
    } else {
        buffersData = CreatePlainBuffers(float4VTangents, geometry_);
    }

    free(float4VTangents);

    const vbdata& vertexData = std::get<0>(buffersData);
    const ibdata& indexData = std::get<1>(buffersData);

    size_t stride = sizeof(BaseVertex);
    VertexBuffer* vbuf = nullptr;
    if (hasUVs) {
        stride = sizeof(TexturedVertex);
    } else if (hasColors) {
        stride = sizeof(ColoredVertex);
    }

    vbuf = BuildFilamentVertexBuffer(engine, vertexData.verticesCount, stride,
                                     hasUVs, hasColors);

    VertexBufferHandle vbHandle;
    if (vbuf) {
        vbHandle = resourceManager.AddVertexBuffer(vbuf);
    } else {
        free(vertexData.bytes);
        free(indexData.bytes);

        return {};
    }

    // Gives ownership of vertexData.bytes to VertexBuffer, which will
    // be deallocated later with DeallocateBuffer.
    VertexBuffer::BufferDescriptor vertexbufferDescriptor(
            vertexData.bytes, vertexData.bytesToCopy);
    vertexbufferDescriptor.setCallback(
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vertexbufferDescriptor));

    auto ibHandle = resourceManager.CreateIndexBuffer(
            indexData.bytesCount / indexData.stride, indexData.stride);
    auto ibuf = resourceManager.GetIndexBuffer(ibHandle).lock();

    // Gives ownership of indexData.bytes to IndexBuffer, which will
    // be deallocated later with DeallocateBuffer.
    IndexBuffer::BufferDescriptor indicesDescriptor(indexData.bytes,
                                                    indexData.bytesCount);
    indicesDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));

    return std::make_tuple(vbHandle, ibHandle);
}

filament::Box TriangleMeshBuffersBuilder::ComputeAABB() {
    auto geometryAABB = geometry_.GetAxisAlignedBoundingBox();

    const filament::math::float3 min(geometryAABB.min_bound_.x(),
                                     geometryAABB.min_bound_.y(),
                                     geometryAABB.min_bound_.z());
    const filament::math::float3 max(geometryAABB.max_bound_.x(),
                                     geometryAABB.max_bound_.y(),
                                     geometryAABB.max_bound_.z());

    Box aabb;
    aabb.set(min, max);

    return aabb;
}

}  // namespace visualization
}  // namespace open3d
