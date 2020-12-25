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

#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace visualization {
namespace rendering {

class TemporaryLineSetBuilder : public LineSetBuffersBuilder {
public:
    explicit TemporaryLineSetBuilder(std::shared_ptr<geometry::LineSet> lines)
        : LineSetBuffersBuilder(*lines), lines_(lines) {}

private:
    std::shared_ptr<geometry::LineSet> lines_;
};

std::unique_ptr<GeometryBuffersBuilder> GeometryBuffersBuilder::GetBuilder(
        const geometry::Geometry3D& geometry) {
    using GT = geometry::Geometry::GeometryType;

    switch (geometry.GetGeometryType()) {
        case GT::TriangleMesh:
            return std::make_unique<TriangleMeshBuffersBuilder>(
                    static_cast<const geometry::TriangleMesh&>(geometry));

        case GT::PointCloud:
            return std::make_unique<PointCloudBuffersBuilder>(
                    static_cast<const geometry::PointCloud&>(geometry));

        case GT::LineSet:
            return std::make_unique<LineSetBuffersBuilder>(
                    static_cast<const geometry::LineSet&>(geometry));
        case GT::OrientedBoundingBox: {
            auto obb =
                    static_cast<const geometry::OrientedBoundingBox&>(geometry);
            auto lines = geometry::LineSet::CreateFromOrientedBoundingBox(obb);
            lines->PaintUniformColor(obb.color_);
            return std::make_unique<TemporaryLineSetBuilder>(lines);
        }
        case GT::AxisAlignedBoundingBox: {
            auto aabb = static_cast<const geometry::AxisAlignedBoundingBox&>(
                    geometry);
            auto lines =
                    geometry::LineSet::CreateFromAxisAlignedBoundingBox(aabb);
            lines->PaintUniformColor(aabb.color_);
            return std::make_unique<TemporaryLineSetBuilder>(lines);
        }
        default:
            break;
    }

    return nullptr;
}

std::unique_ptr<GeometryBuffersBuilder> GeometryBuffersBuilder::GetBuilder(
        const t::geometry::PointCloud& geometry) {
    return std::make_unique<TPointCloudBuffersBuilder>(geometry);
}

void GeometryBuffersBuilder::DeallocateBuffer(void* buffer,
                                              size_t size,
                                              void* user_ptr) {
    free(buffer);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
