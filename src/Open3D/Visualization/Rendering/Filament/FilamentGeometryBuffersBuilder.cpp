// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
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

std::unique_ptr<GeometryBuffersBuilder> GeometryBuffersBuilder::GetBuilder(const geometry::Geometry3D& geometry) {
    using GT = geometry::Geometry::GeometryType;

    switch (geometry.GetGeometryType()) {
        case GT::TriangleMesh:
            return std::make_unique<TriangleMeshBuffersBuilder>(
                    static_cast<const geometry::TriangleMesh&>(geometry));

        case GT::PointCloud:
            return std::make_unique<PointCloudBuffersBuilder>(
                    static_cast<const geometry::PointCloud&>(geometry));
        default:
            break;
    }

    return nullptr;
}

void GeometryBuffersBuilder::DeallocateBuffer(void* buffer, size_t size, void* userPtr) {
    free(buffer);
}

// ===

namespace {
struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 0.f};
    math::float4 color = {1.f, 1.f, 1.f, 1.f};

    static size_t GetPositionOffset() { return offsetof(ColoredVertex, position); }
    static size_t GetTangentOffset() { return offsetof(ColoredVertex, tangent); }
    static size_t GetColorOffset() { return offsetof(ColoredVertex, color); }

    void SetVertexPosition(const Eigen::Vector3d& pos) {
        auto floatPos = pos.cast<float>();
        position.x = floatPos(0);
        position.y = floatPos(1);
        position.z = floatPos(2);
    }

    void SetVertexColor(const Eigen::Vector3d& c) {
        auto floatColor = c.cast<float>();
        color.x = floatColor(0);
        color.y = floatColor(1);
        color.z = floatColor(2);
    }
};
}

PointCloudBuffersBuilder::PointCloudBuffersBuilder(const geometry::PointCloud& geometry)
    : geometry_(geometry) {
}

RenderableManager::PrimitiveType PointCloudBuffersBuilder::GetPrimitiveType() const {
    return RenderableManager::PrimitiveType::POINTS;
}

std::tuple<VertexBufferHandle, IndexBufferHandle> PointCloudBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.points_.size();

    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(nVertices)
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::TANGENTS)
                                 .attribute(VertexAttribute::TANGENTS, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetTangentOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetColorOffset(),
                                            sizeof(ColoredVertex))
                                 .build(engine);

    VertexBufferHandle vbHandle;
    if (vbuf) {
        vbHandle = resourceManager.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = geometry_.normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
    auto float4VTangents = static_cast<math::quatf*>(malloc(tangentsBytesCount));
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
            .vertexCount(nVertices)
            .normals(reinterpret_cast<math::float3*>(normals.data()))
            .build();
    orientation.getQuats(float4VTangents, nVertices);

    size_t verticesBytesCount = nVertices * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(verticesBytesCount));
    for (size_t i = 0; i < geometry_.points_.size(); ++i) {
        ColoredVertex& element = vertices[i];
        element.SetVertexPosition(geometry_.points_[i]);
        element.tangent = float4VTangents[i];
        element.SetVertexColor(geometry_.colors_[i]);
    }

    free(float4VTangents);

    VertexBuffer::BufferDescriptor vertexbufferDescriptor(vertices, verticesBytesCount);
    vertexbufferDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vertexbufferDescriptor));

    size_t indicesBytesCount = nVertices*sizeof(IndexType);
    auto *uintIndices = static_cast<IndexType*>(malloc(indicesBytesCount));
    for (std::uint32_t i = 0; i < nVertices; ++i) {
        uintIndices[i] = i;
    }

    auto ibHandle = resourceManager.CreateIndexBuffer(nVertices, sizeof(IndexType));
    if (!ibHandle) {
        free(uintIndices);
        return {};
    }

    auto ibuf = resourceManager.GetIndexBuffer(ibHandle).lock();

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(uintIndices, indicesBytesCount);
    indicesDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));

    return std::make_tuple(vbHandle, ibHandle);
}

filament::Box PointCloudBuffersBuilder::ComputeAABB() {
    auto geometryAABB = geometry_.GetAxisAlignedBoundingBox();

    const filament::math::float3 min(geometryAABB.min_bound_.x(), geometryAABB.min_bound_.y() ,geometryAABB.min_bound_.z());
    const filament::math::float3 max(geometryAABB.max_bound_.x(), geometryAABB.max_bound_.y() ,geometryAABB.max_bound_.z());

    filament::Box aabb;
    aabb.set(min, max);

    return aabb;
}

}
}