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

#pragma once

#include "Open3D/Visualization/Rendering/RendererHandle.h"

#include <memory>

#include <filament/Box.h>
#include <filament/RenderableManager.h>

namespace open3d {

namespace geometry {
    class Geometry3D;
    class PointCloud;
    class TriangleMesh;
}

namespace visualization {

class GeometryBuffersBuilder {
public:
    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(const geometry::Geometry3D& geometry);
    virtual ~GeometryBuffersBuilder() {}

    virtual filament::RenderableManager::PrimitiveType GetPrimitiveType() const = 0;

    virtual VertexBufferHandle ConstructVertexBuffer() = 0;
    virtual IndexBufferHandle ConstructIndexBuffer() = 0;
    virtual filament::Box ComputeAABB() = 0;
};

class TriangleMeshBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TriangleMeshBuffersBuilder(const geometry::TriangleMesh& geometry);
    ~TriangleMeshBuffersBuilder() override;

    filament::RenderableManager::PrimitiveType GetPrimitiveType() const override;

    VertexBufferHandle ConstructVertexBuffer() override;
    IndexBufferHandle ConstructIndexBuffer() override;
    filament::Box ComputeAABB() override;

private:
    size_t GetIndexStride() const;

    const geometry::TriangleMesh& geometry_;

    filament::math::float3* vertices_ = nullptr;
    size_t verticesBytesCount_ = 0;
    bool freeVertices_ = true;

    std::uint16_t* indices_ = nullptr;
    size_t indicesBytesCount_ = 0;
    bool freeIndices_ = true;
};

class PointCloudBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit PointCloudBuffersBuilder(const geometry::PointCloud& geometry);
    ~PointCloudBuffersBuilder() override;

    filament::RenderableManager::PrimitiveType GetPrimitiveType() const override;

    VertexBufferHandle ConstructVertexBuffer() override;
    IndexBufferHandle ConstructIndexBuffer() override;
    filament::Box ComputeAABB() override;

private:
    using IndexType = std::uint32_t;

    const geometry::PointCloud& geometry_;

    filament::math::float3* vertices_ = nullptr;
    size_t verticesBytesCount_ = 0;
    bool freeVertices_ = true;

    IndexType* indices_ = nullptr;
    size_t indicesBytesCount_ = 0;
    bool freeIndices_ = true;
};

}
}