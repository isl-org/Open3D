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
#include "Open3D/Geometry/LineSet.h"

#include <filament/IndexBuffer.h>
#include <filament/VertexBuffer.h>

#include <Eigen/Core>

#include <map>

using namespace filament;

namespace open3d {
namespace visualization {

namespace {
struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    math::float4 color = {1.f, 1.f, 1.f, 1.f};

    static size_t GetPositionOffset() {
        return offsetof(ColoredVertex, position);
    }
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

}  // namespace

LineSetBuffersBuilder::LineSetBuffersBuilder(const geometry::LineSet& geometry)
    : geometry_(geometry) {}

RenderableManager::PrimitiveType LineSetBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::LINES;
}

LineSetBuffersBuilder::Buffers LineSetBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resourceManager = EngineInstance::GetResourceManager();

    struct LookupKey {
        LookupKey() = default;
        explicit LookupKey(const Eigen::Vector3d& pos,
                           const Eigen::Vector3d& color) {
            values[0] = pos.x();
            values[1] = pos.y();
            values[2] = pos.z();
            values[3] = color.x();
            values[4] = color.y();
            values[5] = color.z();
        }

        // Not necessarily transitive.
        // TODO: does this break sort and map?
        bool operator<(const LookupKey& other) const {
            for (int i = 0; i < 6; ++i) {
                double diff = abs(values[i] - other.values[i]);
                if (diff > kEpsilon) {
                    return values[i] < other.values[i];
                }
            }

            return false;
        }

        const double kEpsilon = 0.00001;
        double values[6] = {0};
    };

    // <source, real>
    std::map<LookupKey, std::pair<GeometryBuffersBuilder::IndexType,
                                  GeometryBuffersBuilder::IndexType>>
            indexLookup;

    const size_t linesCount = geometry_.lines_.size();
    const size_t verticesBytesCount = linesCount * 2 * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(verticesBytesCount));

    const size_t indicesBytesCount = linesCount * 2 * sizeof(IndexType);
    auto* indices = static_cast<IndexType*>(malloc(indicesBytesCount));

    const bool hasColors = geometry_.HasColors();
    Eigen::Vector3d kWhite(1.0, 1.0, 1.0);
    size_t vertexIndex = 0;
    for (size_t i = 0; i < linesCount; ++i) {
        const auto& line = geometry_.lines_[i];

        for (size_t j = 0; j < 2; ++j) {
            size_t index = line(j);

            auto& color = kWhite;
            if (hasColors) {
                color = geometry_.colors_[i];
            }
            const auto& pos = geometry_.points_[index];

            LookupKey lookupKey(pos, color);
            auto found = indexLookup.find(lookupKey);
            if (found != indexLookup.end()) {
                index = found->second.second;
            } else {
                auto& element = vertices[vertexIndex];

                element.SetVertexPosition(pos);
                element.SetVertexColor(color);

                indexLookup[lookupKey] = {index, vertexIndex};
                index = vertexIndex;

                ++vertexIndex;
            }

            indices[2 * i + j] = index;
        }
    }

    const size_t verticesCount = vertexIndex;

    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(verticesCount)
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
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
        free(vertices);
        free(indices);
        return {};
    }

    // Moving `vertices` to VertexBuffer, which will clean them up later
    // with DeallocateBuffer
    VertexBuffer::BufferDescriptor vertexbufferDescriptor(
            vertices, verticesCount * sizeof(ColoredVertex));
    vertexbufferDescriptor.setCallback(
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vertexbufferDescriptor));

    const size_t indicesCount = linesCount * 2;
    auto ibHandle =
            resourceManager.CreateIndexBuffer(indicesCount, sizeof(IndexType));
    if (!ibHandle) {
        free(indices);
        return {};
    }

    auto ibuf = resourceManager.GetIndexBuffer(ibHandle).lock();

    // Moving `indices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    IndexBuffer::BufferDescriptor indicesDescriptor(indices, indicesBytesCount);
    indicesDescriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indicesDescriptor));

    return std::make_tuple(vbHandle, ibHandle);
}

Box LineSetBuffersBuilder::ComputeAABB() {
    const auto geometryAABB = geometry_.GetAxisAlignedBoundingBox();

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
