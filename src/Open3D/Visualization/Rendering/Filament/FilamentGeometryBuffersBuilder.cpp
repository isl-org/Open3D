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

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h"

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/geometry/SurfaceOrientation.h>

using namespace filament;

namespace open3d {
namespace visualization {

static void freeBufferDescriptor(void* buffer, size_t, void*) { free(buffer); }

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

TriangleMeshBuffersBuilder::TriangleMeshBuffersBuilder(const geometry::TriangleMesh& geometry)
    : geometry_(geometry) {
    const size_t nVertices = geometry_.vertices_.size();

    // Copying vertex coordinates
    verticesBytesCount_ = nVertices * 3 * sizeof(float);
    auto* float3VCoordtmp = (Eigen::Vector3f*)malloc(verticesBytesCount_);
    for (size_t i = 0; i < nVertices; ++i) {
        float3VCoordtmp[i] = geometry_.vertices_[i].cast<float>();
    }

    vertices_ = (math::float3*)float3VCoordtmp;

    // Copying indices data
    // FIXME: Potentially memory corruption/misinterpret issue, due to different index stride
    indicesBytesCount_ = geometry_.triangles_.size() * 3 * GetIndexStride();
    auto* uint3Indices = (Eigen::Vector3i*)malloc(indicesBytesCount_);
    for (size_t i = 0; i < geometry_.triangles_.size(); ++i) {
        uint3Indices[i] = geometry_.triangles_[i];
    }

    indices_ = (std::uint16_t*)uint3Indices;
}

TriangleMeshBuffersBuilder::~TriangleMeshBuffersBuilder() {
    if (freeVertices_) {
        free(vertices_);
        vertices_ = nullptr;
    }

    if (freeIndices_) {
        free(indices_);
        indices_ = nullptr;
    }
}

RenderableManager::PrimitiveType TriangleMeshBuffersBuilder::GetPrimitiveType() const {
    return RenderableManager::PrimitiveType::TRIANGLES;
}

VertexBufferHandle TriangleMeshBuffersBuilder::ConstructVertexBuffer() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.vertices_.size();

    VertexBuffer* vbuf =
            VertexBuffer::Builder()
                    .bufferCount(3)
                    .vertexCount(nVertices)
                    .normalized(VertexAttribute::TANGENTS)
                    .normalized(VertexAttribute::COLOR)
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT3, 0)
                    .attribute(VertexAttribute::TANGENTS, 1,
                               VertexBuffer::AttributeType::FLOAT4, 0)
                    .attribute(VertexAttribute::COLOR, 2, VertexBuffer::AttributeType::FLOAT4, 0)
                    .build(engine);

    VertexBufferHandle handle;
    if (vbuf) {
        handle = resourceManager.AddVertexBuffer(vbuf);
    } else {
        return handle;
    }

    // Moving copied vertex coordinates to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor coordsDescriptor(vertices_, verticesBytesCount_);
    coordsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 0, std::move(coordsDescriptor));
    freeVertices_ = false;

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = geometry_.vertex_normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    const size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4VTangents = (math::quatf*)malloc(tangentsBytesCount);
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
            .vertexCount(nVertices)
            .normals((math::float3*)normals.data())
            .build();
    orientation.getQuats(float4VTangents, nVertices);

    // Moving allocated tangents to VertexBuffer
    // they will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor tangentsDescriptor(float4VTangents,
                                                      tangentsBytesCount);
    tangentsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 1, std::move(tangentsDescriptor));

    // Copying colors
    size_t colorsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4Colors = (math::float4*)malloc(colorsBytesCount);
    if (geometry_.vertex_colors_.empty()) {
        for (size_t i = 0; i < nVertices; ++i) {
            float4Colors[i] = {1.f,1.f,1.f,1.f};
        }
    } else {
        for (size_t i = 0; i < nVertices; ++i) {
            auto c = geometry_.vertex_colors_[i];

            float4Colors[i].r = c.x();
            float4Colors[i].g = c.y();
            float4Colors[i].b = c.z();
            float4Colors[i].a = 1.f;
        }
    }

    // Moving colors to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor colorsDescriptor(float4Colors,
                                                    colorsBytesCount);
    colorsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 2, std::move(colorsDescriptor));

    return handle;
}

IndexBufferHandle TriangleMeshBuffersBuilder::ConstructIndexBuffer() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    auto handle = resourceManager.CreateIndexBuffer(
            geometry_.triangles_.size() * 3, GetIndexStride());

    auto ibuf = resourceManager.GetIndexBuffer(handle).lock();

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(indices_, indicesBytesCount_);
    indicesDescriptor.setCallback(freeBufferDescriptor);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));
    freeIndices_ = false;

    return handle;
}

filament::Box TriangleMeshBuffersBuilder::ComputeAABB() {
    const size_t nVertices = geometry_.vertices_.size();

    Box aabb;
    if (GetIndexStride() == sizeof(std::uint16_t)) {
        aabb = RenderableManager::computeAABB(vertices_,
                                              indices_,
                                              nVertices);
    } else {
        // FIXME: Potentially memory corruption/misinterpret issue, due to different index stride
        aabb = RenderableManager::computeAABB(vertices_,
                                              (std::uint32_t*)indices_,
                                              nVertices);
    }

    return aabb;
}

size_t TriangleMeshBuffersBuilder::GetIndexStride() const {
    return sizeof(geometry_.triangles_[0][0]);
}

// ===

PointCloudBuffersBuilder::PointCloudBuffersBuilder(const geometry::PointCloud& geometry)
    : geometry_(geometry) {
    const size_t nVertices = geometry_.points_.size();

    // Copying vertex coordinates
    verticesBytesCount_ = nVertices * 3 * sizeof(float);
    auto* float3VCoordtmp = (Eigen::Vector3f*)malloc(verticesBytesCount_);
    for (size_t i = 0; i < nVertices; ++i) {
        float3VCoordtmp[i] = geometry_.points_[i].cast<float>();
    }

    vertices_ = (math::float3*)float3VCoordtmp;

    // Creating indices data
    indicesBytesCount_ = nVertices*sizeof(IndexType);
    auto *uintIndices = (IndexType*)malloc(indicesBytesCount_);
    for (IndexType i = 0; i < nVertices; ++i) {
        uintIndices[i] = i;
    }

    indices_ = uintIndices;
}

PointCloudBuffersBuilder::~PointCloudBuffersBuilder() {
    if (freeVertices_) {
        free(vertices_);
        vertices_ = nullptr;
    }

    if (freeIndices_) {
        free(indices_);
        indices_ = nullptr;
    }
}

RenderableManager::PrimitiveType PointCloudBuffersBuilder::GetPrimitiveType() const {
    return RenderableManager::PrimitiveType::POINTS;
}

VertexBufferHandle PointCloudBuffersBuilder::ConstructVertexBuffer() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.points_.size();

    VertexBuffer* vbuf =
            VertexBuffer::Builder()
                    .bufferCount(3)
                    .vertexCount(nVertices)
                    .normalized(VertexAttribute::TANGENTS)
                    .normalized(VertexAttribute::COLOR)
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT3, 0)
                    .attribute(VertexAttribute::TANGENTS, 1,
                               VertexBuffer::AttributeType::FLOAT4, 0)
                    .attribute(VertexAttribute::COLOR, 2,
                               VertexBuffer::AttributeType::FLOAT4, 0)
                    .build(engine);

    VertexBufferHandle handle;
    if (vbuf) {
        handle = resourceManager.AddVertexBuffer(vbuf);
    } else {
        return handle;
    }

    // Moving copied vertex coordinates to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor coordsDescriptor(vertices_,
                                                    verticesBytesCount_);
    coordsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 0, std::move(coordsDescriptor));
    freeVertices_ = false;

    // Converting vertex normals to float base
    std::vector<Eigen::Vector3f> normals;
    normals.resize(nVertices);
    for (size_t i = 0; i < nVertices; ++i) {
        normals[i] = geometry_.normals_[i].cast<float>();
    }

    // Converting normals to Filament type - quaternions
    size_t tangentsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4VTangents = (math::quatf*)malloc(tangentsBytesCount);
    auto orientation = filament::geometry::SurfaceOrientation::Builder()
                               .vertexCount(nVertices)
                               .normals((math::float3*)normals.data())
                               .build();
    orientation.getQuats(float4VTangents, nVertices);

    // Moving allocated tangents to VertexBuffer
    // they will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor tangentsDescriptor(float4VTangents,
                                                      tangentsBytesCount);
    tangentsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 1, std::move(tangentsDescriptor));

    // Copying colors
    size_t colorsBytesCount = nVertices * 4 * sizeof(float);
    auto* float4Colors = (math::float4*)malloc(colorsBytesCount);
    for (size_t i = 0; i < nVertices; ++i) {
        auto c = geometry_.colors_[i];
        float4Colors[i].r = c.x();
        float4Colors[i].g = c.y();
        float4Colors[i].b = c.z();
        float4Colors[i].a = 1.f;
    }

    // Moving colors to VertexBuffer
    // malloc'ed memory will be freed later with freeBufferDescriptor
    VertexBuffer::BufferDescriptor colorsDescriptor(float4Colors,
                                                    colorsBytesCount);
    colorsDescriptor.setCallback(freeBufferDescriptor);
    vbuf->setBufferAt(engine, 2, std::move(colorsDescriptor));

    return handle;
}

IndexBufferHandle PointCloudBuffersBuilder::ConstructIndexBuffer() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    const size_t nVertices = geometry_.points_.size();

    auto handle = resourceManager.CreateIndexBuffer(nVertices, sizeof(IndexType));
    if (!handle) {
        return handle;
    }

    auto ibuf = resourceManager.GetIndexBuffer(handle).lock();

    // Moving copied indices to IndexBuffer
    // they will be freed later with freeBufferDescriptor
    IndexBuffer::BufferDescriptor indicesDescriptor(indices_, indicesBytesCount_);
    indicesDescriptor.setCallback(freeBufferDescriptor);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));
    freeIndices_ = false;

    return handle;
}

filament::Box PointCloudBuffersBuilder::ComputeAABB() {
    const size_t nVertices = geometry_.points_.size();

    return RenderableManager::computeAABB(vertices_, (IndexType*)indices_, nVertices);
}

}
}