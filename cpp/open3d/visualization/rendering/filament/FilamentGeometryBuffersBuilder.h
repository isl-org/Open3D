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

// NOTE: This header must precede the Filament headers otherwise a conflict
// occurs between Filament and standard headers
#include "open3d/visualization/rendering/RendererHandle.h"

#include <filament/Box.h>
#include <filament/RenderableManager.h>
#include <memory>
#include <tuple>

namespace open3d {

namespace geometry {
class Geometry3D;
class LineSet;
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace visualization {
namespace rendering {

class GeometryBuffersBuilder {
public:
    using Buffers = std::tuple<VertexBufferHandle, IndexBufferHandle>;
    using IndexType = std::uint32_t;

    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(
            const geometry::Geometry3D& geometry);
    virtual ~GeometryBuffersBuilder() = default;

    virtual filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const = 0;

    virtual Buffers ConstructBuffers() = 0;
    virtual filament::Box ComputeAABB() = 0;

protected:
    static void DeallocateBuffer(void* buffer, size_t size, void* user_ptr);
};

class TriangleMeshBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TriangleMeshBuffersBuilder(const geometry::TriangleMesh& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const geometry::TriangleMesh& geometry_;
};

class PointCloudBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit PointCloudBuffersBuilder(const geometry::PointCloud& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const geometry::PointCloud& geometry_;
};

class LineSetBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit LineSetBuffersBuilder(const geometry::LineSet& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const geometry::LineSet& geometry_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
