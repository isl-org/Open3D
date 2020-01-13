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

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h"

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/geometry/SurfaceOrientation.h>

#include <map>

using namespace filament;

namespace open3d {
namespace visualization {

namespace {

struct BaseVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 0.f};
};

struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 0.f};
    math::float4 color = {1.f, 1.f, 1.f, 1.f};
};

struct TexturedVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 0.f};
    math::float4 color = {1.f, 1.f, 1.f, 1.f};
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
size_t GetVertexPositionOffset() { return 0; }

template <typename VertexType>
size_t GetVertexTangentOffset() { return offsetof(VertexType, tangent); }

template <typename VertexType>
size_t GetVertexColorOffset() { return offsetof(VertexType, color); }

template <typename VertexType>
size_t GetVertexUVOffset() { return offsetof(VertexType, uv); }

template <typename VertexType>
size_t GetVertexStride() { return sizeof(VertexType); }

VertexBuffer* BuildFilamentVertexBuffer(filament::Engine& engine, const std::uint32_t verticesCount, const std::uint32_t stride, bool hasUvs, bool hasColors) {
    auto builder = VertexBuffer::Builder()
            .bufferCount(1)
            .vertexCount(verticesCount)
            .attribute(VertexAttribute::POSITION, 0,
                       VertexBuffer::AttributeType::FLOAT3,
                       GetVertexPositionOffset<TexturedVertex>(), stride)
            .normalized(VertexAttribute::TANGENTS)
            .attribute(VertexAttribute::TANGENTS, 0,
                       VertexBuffer::AttributeType::FLOAT4,
                       GetVertexTangentOffset<TexturedVertex>(), stride);

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

std::tuple<vbdata, ibdata> CreatePlainBuffers(const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

    vertexData.verticesCount = geometry.vertices_.size();
    vertexData.bytesCount = vertexData.verticesCount * sizeof(BaseVertex);
    vertexData.bytesToCopy = vertexData.bytesCount;
    vertexData.bytes = malloc(vertexData.bytesCount);

    auto plainVertices = static_cast<BaseVertex*>(vertexData.bytes);
    for (size_t i = 0; i < vertexData.verticesCount; ++i) {
        BaseVertex& element = plainVertices[i];

        SetVertexPosition(element, geometry.vertices_[i]);
        element.tangent = tangents[i];
    }

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(malloc(indexData.bytesCount));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        indexData.bytes[3*i] = triangle(0);
        indexData.bytes[3*i+1] = triangle(1);
        indexData.bytes[3*i+2] = triangle(2);
    }

    return std::make_tuple(vertexData, indexData);
}

std::tuple<vbdata, ibdata> CreateColoredBuffers(const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

    vertexData.verticesCount = geometry.vertices_.size();
    vertexData.bytesCount = vertexData.verticesCount * sizeof(ColoredVertex);
    vertexData.bytesToCopy = vertexData.bytesCount;
    vertexData.bytes = malloc(vertexData.bytesCount);

    auto coloredVertices = static_cast<ColoredVertex*>(vertexData.bytes);
    for (size_t i = 0; i < vertexData.verticesCount; ++i) {
        ColoredVertex& element = coloredVertices[i];

        SetVertexPosition(element, geometry.vertices_[i]);
        element.tangent = tangents[i];
        SetVertexColor(element, geometry.vertex_colors_[i]);
    }

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(malloc(indexData.bytesCount));
    for (size_t i = 0; i < geometry.triangles_.size(); ++i) {
        const auto& triangle = geometry.triangles_[i];
        indexData.bytes[3*i] = triangle(0);
        indexData.bytes[3*i+1] = triangle(1);
        indexData.bytes[3*i+2] = triangle(2);
    }

    return std::make_tuple(vertexData, indexData);
}

std::tuple<vbdata, ibdata> CreateTexturedBuffers(const math::quatf* tangents, const geometry::TriangleMesh& geometry) {
    vbdata vertexData;
    ibdata indexData;

    struct LookupKey {
        LookupKey() = default;
        explicit LookupKey(const Eigen::Vector3d& pos, const Eigen::Vector2d& uv) {
            values[0] = pos.x();
            values[1] = pos.y();
            values[2] = pos.z();
            values[3] = uv.x();
            values[4] = uv.y();
        }

        bool operator<(const LookupKey& other) const {
            for (std::uint8_t i = 0; i < 5; ++i) {
                double diff = abs(values[i] - other.values[i]);
                if (diff > kEpsilon && values[i] < other.values[i]) {
                    return true;
                }
            }

            return false;
        }

        const double kEpsilon = 0.00001;
        double values[5] = {0};
    };
    //                           < real index  , source index >
    std::map<LookupKey, std::pair<GeometryBuffersBuilder::IndexType, GeometryBuffersBuilder::IndexType>> indexLookup;

    indexData.stride = sizeof(GeometryBuffersBuilder::IndexType);
    indexData.bytesCount = geometry.triangles_.size() * 3 * indexData.stride;
    indexData.bytes = static_cast<GeometryBuffersBuilder::IndexType*>(malloc(indexData.bytesCount));

    vertexData.bytesCount = geometry.triangles_.size() * 3 * sizeof(TexturedVertex);
    vertexData.bytes = malloc(vertexData.bytesCount);

    GeometryBuffersBuilder::IndexType freeIndex = 0;
    GeometryBuffersBuilder::IndexType uvIndex = 0;
    auto texturedVertices = static_cast<TexturedVertex*>(vertexData.bytes);

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
                element.tangent = tangents[sourceIndex];
                SetVertexColor(element, geometry.vertex_colors_[sourceIndex]);
                SetVertexUV(element, uv);
            }

            indexData.bytes[3 * i + j] = index;

            ++uvIndex;
        }
    }

    vertexData.verticesCount = freeIndex;
    vertexData.bytesToCopy = vertexData.verticesCount * sizeof(TexturedVertex);

    return std::make_tuple(vertexData, indexData);
}

}

TriangleMeshBuffersBuilder::TriangleMeshBuffersBuilder(
        const geometry::TriangleMesh& geometry)
    : geometry_(geometry) {
}

RenderableManager::PrimitiveType TriangleMeshBuffersBuilder::GetPrimitiveType() const {
    return RenderableManager::PrimitiveType::TRIANGLES;
}

std::tuple<VertexBufferHandle, IndexBufferHandle> TriangleMeshBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.vertices_.size();

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = geometry_.vertex_normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    const size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4VTangents = static_cast<math::quatf*>(malloc(tangentsBytesCount));
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
            .vertexCount(nVertices)
            .normals(reinterpret_cast<math::float3*>(normals.data()))
            .build();
    orientation.getQuats(float4VTangents, nVertices);

    const bool hasColors = geometry_.HasVertexColors();
    const bool hasUVs = geometry_.HasTriangleUvs();

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

    vbuf = BuildFilamentVertexBuffer(engine, vertexData.verticesCount, stride, hasUVs, hasColors);

    VertexBufferHandle vbHandle;
    if (vbuf) {
        vbHandle = resourceManager.AddVertexBuffer(vbuf);
    } else {
        free(vertexData.bytes);
        free(indexData.bytes);

        return {};
    }

    VertexBuffer::BufferDescriptor vertexbufferDescriptor(
            vertexData.bytes, vertexData.bytesToCopy);
    vertexbufferDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vertexbufferDescriptor));

    auto ibHandle = resourceManager.CreateIndexBuffer(
            indexData.bytesCount/ indexData.stride, indexData.stride);
    auto ibuf = resourceManager.GetIndexBuffer(ibHandle).lock();

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(indexData.bytes,
                                                    indexData.bytesCount);
    indicesDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));

    return std::make_tuple(vbHandle, ibHandle);
}

filament::Box TriangleMeshBuffersBuilder::ComputeAABB() {
    auto geometryAABB = geometry_.GetAxisAlignedBoundingBox();

    const filament::math::float3 min(geometryAABB.min_bound_.x(), geometryAABB.min_bound_.y() ,geometryAABB.min_bound_.z());
    const filament::math::float3 max(geometryAABB.max_bound_.x(), geometryAABB.max_bound_.y() ,geometryAABB.max_bound_.z());

    filament::Box aabb;
    aabb.set(min, max);

    return aabb;
}

}
}