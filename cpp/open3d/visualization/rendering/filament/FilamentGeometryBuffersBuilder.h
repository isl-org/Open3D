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

// clang-format off
// NOTE: This header must precede the Filament headers otherwise a conflict
// occurs between Filament and standard headers
#include "open3d/visualization/rendering/RendererHandle.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293:  Filament's utils/algorithm.h utils::details::clz() does strange
//        things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//        32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//        determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif // _MSC_VER

#include <filament/Box.h>
#include <filament/RenderableManager.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
// clang-format on

#include <memory>
#include <tuple>

namespace open3d {

namespace geometry {
class Geometry3D;
class LineSet;
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace t {
namespace geometry {
class PointCloud;
}
}  // namespace t

namespace visualization {
namespace rendering {

class GeometryBuffersBuilder {
public:
    // Note that the downsampled index buffer may be kBadId if a downsampled
    // buffer was not requested, failed, or cannot be created (e.g. if not
    // a point cloud).
    using Buffers = std::tuple<VertexBufferHandle,  // vertex buffer
                               IndexBufferHandle,   // index buffer
                               IndexBufferHandle>;  // downsampled buffer
    using IndexType = std::uint32_t;

    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(
            const geometry::Geometry3D& geometry);
    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(
            const t::geometry::PointCloud& geometry);

    virtual ~GeometryBuffersBuilder() = default;

    virtual filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const = 0;

    // Defaults to infinity (that is, no downsampling). If threshold is
    // set and the number of points exceeds the threshold, ConstructBuffers()
    // will return a downsampled index buffer. Certain builders may ignore
    // this threshold.
    virtual void SetDownsampleThreshold(size_t min_points) {
        downsample_threshold_ = min_points;
    }

    virtual Buffers ConstructBuffers() = 0;
    virtual filament::Box ComputeAABB() = 0;

protected:
    size_t downsample_threshold_ = SIZE_MAX;

    static void DeallocateBuffer(void* buffer, size_t size, void* user_ptr);

    static IndexBufferHandle CreateIndexBuffer(size_t max_index,
                                               size_t n_subsamples = SIZE_MAX);
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

class TPointCloudBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TPointCloudBuffersBuilder(const t::geometry::PointCloud& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const t::geometry::PointCloud& geometry_;
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
