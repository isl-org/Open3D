// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <filament/IndexBuffer.h>
#include <filament/VertexBuffer.h>
#include <geometry/SurfaceOrientation.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

using namespace filament;

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
struct ColoredVertex {
    math::float3 position = {0.f, 0.f, 0.f};
    // Default to mid-gray which provides good separation from the two most
    // common background colors: white and black. Otherwise, point clouds
    // without per-vertex colors may appear is if they are not rendering because
    // they blend with the background.
    math::float4 color = {0.5f, 0.5f, 0.5f, 1.f};
    math::quatf tangent = {0.f, 0.f, 0.f, 1.f};
    math::float2 uv = {0.f, 0.f};

    static std::uint32_t GetPositionOffset() {
        return offsetof(ColoredVertex, position);
    }
    static std::uint32_t GetColorOffset() {
        return offsetof(ColoredVertex, color);
    }
    static std::uint32_t GetTangentOffset() {
        return offsetof(ColoredVertex, tangent);
    }
    static std::uint32_t GetUVOffset() { return offsetof(ColoredVertex, uv); }
    void SetVertexPosition(const Eigen::Vector3d& pos) {
        auto float_pos = pos.cast<float>();
        position.x = float_pos(0);
        position.y = float_pos(1);
        position.z = float_pos(2);
    }

    float sRGBToLinear(float color) {
        return color <= 0.04045f ? color / 12.92f
                                 : pow((color + 0.055f) / 1.055f, 2.4f);
    }

    void SetVertexColor(const Eigen::Vector3d& c, bool adjust_for_srgb) {
        auto float_color = c.cast<float>();
        if (adjust_for_srgb) {
            color.x = sRGBToLinear(float_color(0));
            color.y = sRGBToLinear(float_color(1));
            color.z = sRGBToLinear(float_color(2));
        } else {
            color.x = float_color(0);
            color.y = float_color(1);
            color.z = float_color(2);
        }
    }
};
}  // namespace

IndexBufferHandle GeometryBuffersBuilder::CreateIndexBuffer(
        size_t max_index, size_t n_subsamples /*= SIZE_MAX*/) {
    using IndexType = GeometryBuffersBuilder::IndexType;
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    size_t n_indices = std::min(max_index, n_subsamples);
    // Use double for precision, since float can only accurately represent
    // integers up to 2^24 = 16 million, and we have buffers with more points
    // than that.
    double step = double(max_index) / double(n_indices);

    const size_t n_bytes = n_indices * sizeof(IndexType);
    auto* uint_indices = static_cast<IndexType*>(malloc(n_bytes));
    if (step <= 1.0) {
        // std::iota is about 2X faster than a loop on my machine, anyway.
        // Since this is the common case, and is used for every entity,
        // special-case this to make it fast.
        std::iota(uint_indices, uint_indices + n_indices, 0);
    } else if (std::floor(step) == step) {
        for (size_t i = 0; i < n_indices; ++i) {
            uint_indices[i] = IndexType(step * i);
        }
    } else {
        size_t idx = 0;
        uint_indices[idx++] = 0;
        double dist = 1.0;
        size_t i;
        for (i = 1; i < max_index; ++i) {
            if (dist >= step) {
                uint_indices[idx++] = IndexType(i);
                dist -= step;
                if (idx > n_indices) {  // paranoia, should not happen
                    break;
                }
            }
            dist += 1.0;
        }
        // Very occasionally floating point error leads to one fewer points
        // being added.
        if (i < max_index - 1) {
            n_indices = i + 1;
        }
    }

    auto ib_handle =
            resource_mgr.CreateIndexBuffer(n_indices, sizeof(IndexType));
    if (!ib_handle) {
        free(uint_indices);
        return IndexBufferHandle();
    }

    auto ibuf = resource_mgr.GetIndexBuffer(ib_handle).lock();

    // Moving `uintIndices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    IndexBuffer::BufferDescriptor indices_descriptor(uint_indices, n_bytes);
    indices_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    ibuf->setBuffer(engine, std::move(indices_descriptor));
    return ib_handle;
}

PointCloudBuffersBuilder::PointCloudBuffersBuilder(
        const geometry::PointCloud& geometry)
    : geometry_(geometry) {}

RenderableManager::PrimitiveType PointCloudBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::POINTS;
}

GeometryBuffersBuilder::Buffers PointCloudBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const size_t n_vertices = geometry_.points_.size();

    // We use CUSTOM0 for tangents along with TANGENTS attribute
    // because Filament would optimize out anything about normals and lightning
    // from unlit materials. But our shader for normals visualizing is unlit, so
    // we need to use this workaround.
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(1)
                                 .vertexCount(std::uint32_t(n_vertices))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3,
                                            ColoredVertex::GetPositionOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetColorOffset(),
                                            sizeof(ColoredVertex))
                                 .normalized(VertexAttribute::TANGENTS)
                                 .attribute(VertexAttribute::TANGENTS, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetTangentOffset(),
                                            sizeof(ColoredVertex))
                                 .attribute(VertexAttribute::CUSTOM0, 0,
                                            VertexBuffer::AttributeType::FLOAT4,
                                            ColoredVertex::GetTangentOffset(),
                                            sizeof(ColoredVertex))
                                 .attribute(VertexAttribute::UV0, 0,
                                            VertexBuffer::AttributeType::FLOAT2,
                                            ColoredVertex::GetUVOffset(),
                                            sizeof(ColoredVertex))
                                 .build(engine);

    VertexBufferHandle vb_handle;
    if (vbuf) {
        vb_handle = resource_mgr.AddVertexBuffer(vbuf);
    } else {
        return {};
    }

    math::quatf* float4v_tagents = nullptr;
    if (geometry_.HasNormals()) {
        // Converting vertex normals to float base
        std::vector<Eigen::Vector3f> normals;
        normals.resize(n_vertices);
        for (size_t i = 0; i < n_vertices; ++i) {
            normals[i] = geometry_.normals_[i].cast<float>();
        }

        // Converting normals to Filament type - quaternions
        const size_t tangents_byte_count = n_vertices * 4 * sizeof(float);
        float4v_tagents =
                static_cast<math::quatf*>(malloc(tangents_byte_count));
        auto orientation = filament::geometry::SurfaceOrientation::Builder()
                                   .vertexCount(n_vertices)
                                   .normals(reinterpret_cast<math::float3*>(
                                           normals.data()))
                                   .build();
        orientation->getQuats(float4v_tagents, n_vertices);
        delete orientation;
    }

    const size_t vertices_byte_count = n_vertices * sizeof(ColoredVertex);
    auto* vertices = static_cast<ColoredVertex*>(malloc(vertices_byte_count));
    const ColoredVertex kDefault;
    for (size_t i = 0; i < geometry_.points_.size(); ++i) {
        ColoredVertex& element = vertices[i];
        element.SetVertexPosition(geometry_.points_[i]);
        if (geometry_.HasColors()) {
            element.SetVertexColor(geometry_.colors_[i],
                                   adjust_colors_for_srgb_tonemapping_);
        } else {
            element.color = kDefault.color;
        }

        if (float4v_tagents) {
            element.tangent = float4v_tagents[i];
        } else {
            element.tangent = kDefault.tangent;
        }
        element.uv = kDefault.uv;
    }

    free(float4v_tagents);

    // Moving `vertices` to IndexBuffer, which will clean them up later
    // with DeallocateBuffer
    VertexBuffer::BufferDescriptor vb_descriptor(vertices, vertices_byte_count);
    vb_descriptor.setCallback(GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(vb_descriptor));

    auto ib_handle = CreateIndexBuffer(n_vertices);

    IndexBufferHandle downsampled_handle;
    if (n_vertices >= downsample_threshold_) {
        downsampled_handle =
                CreateIndexBuffer(n_vertices, downsample_threshold_);
    }

    return std::make_tuple(vb_handle, ib_handle, downsampled_handle);
}

filament::Box PointCloudBuffersBuilder::ComputeAABB() {
    const auto geometry_aabb = geometry_.GetAxisAlignedBoundingBox();
    filament::math::float3 min(geometry_aabb.min_bound_.x(),
                               geometry_aabb.min_bound_.y(),
                               geometry_aabb.min_bound_.z());
    filament::math::float3 max(geometry_aabb.max_bound_.x(),
                               geometry_aabb.max_bound_.y(),
                               geometry_aabb.max_bound_.z());
    // Filament chokes on empty bounding boxes so don't allow it
    if (geometry_aabb.IsEmpty()) {
        min.x -= 0.1f;
        min.y -= 0.1f;
        min.z -= 0.1f;
        max.x += 0.1f;
        max.y += 0.1f;
        max.z += 0.1f;
    }
    Box aabb;
    aabb.set(min, max);

    return aabb;
}

TPointCloudBuffersBuilder::TPointCloudBuffersBuilder(
        const t::geometry::PointCloud& geometry)
    : geometry_(geometry) {
    // Make sure geometry is on CPU
    auto pts = geometry.GetPointPositions();
    if (pts.GetDevice().GetType() == core::Device::DeviceType::CUDA) {
        utility::LogWarning(
                "GPU resident point clouds are not currently supported for "
                "visualization. Copying data to CPU.");
        geometry_ = geometry.To(core::Device("CPU:0"));
    }

    // Now make sure data types are Float32
    if (pts.GetDtype() != core::Float32) {
        utility::LogWarning(
                "Tensor point cloud points must have DType of Float32 not {}. "
                "Converting.",
                pts.GetDtype().ToString());
        geometry_.GetPointPositions() = pts.To(core::Float32);
    }
    if (geometry_.HasPointNormals() &&
        geometry_.GetPointNormals().GetDtype() != core::Float32) {
        auto normals = geometry_.GetPointNormals();
        utility::LogWarning(
                "Tensor point cloud normals must have DType of Float32 not {}. "
                "Converting.",
                normals.GetDtype().ToString());
        geometry_.GetPointNormals() = normals.To(core::Float32);
    }
    if (geometry_.HasPointColors() &&
        geometry_.GetPointColors().GetDtype() != core::Float32) {
        auto colors = geometry_.GetPointColors();
        utility::LogWarning(
                "Tensor point cloud colors must have DType of Float32 not {}. "
                "Converting.",
                colors.GetDtype().ToString());
        geometry_.GetPointColors() = colors.To(core::Float32);
        if (colors.GetDtype() == core::UInt8) {
            geometry_.GetPointColors() = geometry_.GetPointColors() / 255.0f;
        }
    }
}

RenderableManager::PrimitiveType TPointCloudBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::POINTS;
}

GeometryBuffersBuilder::Buffers TPointCloudBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const auto& points = geometry_.GetPointPositions();
    const size_t n_vertices = points.GetLength();

    // We use CUSTOM0 for tangents along with TANGENTS attribute
    // because Filament would optimize out anything about normals and lightning
    // from unlit materials. But our shader for normals visualizing is unlit, so
    // we need to use this workaround.
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(4)
                                 .vertexCount(uint32_t(n_vertices))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 1,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .normalized(VertexAttribute::TANGENTS)
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

    const size_t vertex_array_size = n_vertices * 3 * sizeof(float);
    float* vertex_array = static_cast<float*>(malloc(vertex_array_size));
    memcpy(vertex_array, points.GetDataPtr(), vertex_array_size);
    VertexBuffer::BufferDescriptor pts_descriptor(
            vertex_array, vertex_array_size,
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 0, std::move(pts_descriptor));

    const size_t color_array_size = n_vertices * 3 * sizeof(float);
    if (geometry_.HasPointColors()) {
        float* color_array = static_cast<float*>(malloc(color_array_size));
        memcpy(color_array, geometry_.GetPointColors().GetDataPtr(),
               color_array_size);
        VertexBuffer::BufferDescriptor color_descriptor(
                color_array, color_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 1, std::move(color_descriptor));
    } else {
        float* color_array = static_cast<float*>(malloc(color_array_size));
        for (size_t i = 0; i < n_vertices * 3; ++i) {
            color_array[i] = 1.f;
        }
        VertexBuffer::BufferDescriptor color_descriptor(
                color_array, color_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 1, std::move(color_descriptor));
    }

    const size_t normal_array_size = n_vertices * 4 * sizeof(float);
    if (geometry_.HasPointNormals()) {
        const auto& normals = geometry_.GetPointNormals();

        // Converting normals to Filament type - quaternions
        auto float4v_tangents =
                static_cast<math::quatf*>(malloc(normal_array_size));
        auto orientation =
                filament::geometry::SurfaceOrientation::Builder()
                        .vertexCount(n_vertices)
                        .normals(reinterpret_cast<const math::float3*>(
                                normals.GetDataPtr()))
                        .build();
        orientation->getQuats(float4v_tangents, n_vertices);
        VertexBuffer::BufferDescriptor normals_descriptor(
                float4v_tangents, normal_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 2, std::move(normals_descriptor));
        delete orientation;
    } else {
        float* normal_array = static_cast<float*>(malloc(normal_array_size));
        float* normal_ptr = normal_array;
        for (size_t i = 0; i < n_vertices; ++i) {
            *normal_ptr++ = 0.f;
            *normal_ptr++ = 0.f;
            *normal_ptr++ = 0.f;
            *normal_ptr++ = 1.f;
        }
        VertexBuffer::BufferDescriptor normals_descriptor(
                normal_array, normal_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 2, std::move(normals_descriptor));
    }

    const size_t uv_array_size = n_vertices * 2 * sizeof(float);
    float* uv_array = static_cast<float*>(malloc(uv_array_size));
    if (geometry_.HasPointAttr("uv")) {
        const float* uv_src = static_cast<const float*>(
                geometry_.GetPointAttr("uv").GetDataPtr());
        memcpy(uv_array, uv_src, uv_array_size);
    } else if (geometry_.HasPointAttr("__visualization_scalar")) {
        // Update in FilamentScene::UpdateGeometry(), too.
        memset(uv_array, 0, uv_array_size);
        auto vis_scalars =
                geometry_.GetPointAttr("__visualization_scalar").Contiguous();
        const float* src = static_cast<const float*>(vis_scalars.GetDataPtr());
        const size_t n = 2 * n_vertices;
        for (size_t i = 0; i < n; i += 2) {
            uv_array[i] = *src++;
        }
    } else {
        memset(uv_array, 0, uv_array_size);
    }
    VertexBuffer::BufferDescriptor uv_descriptor(
            uv_array, uv_array_size, GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 3, std::move(uv_descriptor));

    auto ib_handle = CreateIndexBuffer(n_vertices);

    IndexBufferHandle downsampled_handle;
    if (n_vertices >= downsample_threshold_) {
        downsampled_handle =
                CreateIndexBuffer(n_vertices, downsample_threshold_);
    }

    return std::make_tuple(vb_handle, ib_handle, downsampled_handle);
}

filament::Box TPointCloudBuffersBuilder::ComputeAABB() {
    auto min_bounds = geometry_.GetMinBound();
    auto max_bounds = geometry_.GetMaxBound();
    auto* min_bounds_float = min_bounds.GetDataPtr<float>();
    auto* max_bounds_float = max_bounds.GetDataPtr<float>();

    filament::math::float3 min(min_bounds_float[0], min_bounds_float[1],
                               min_bounds_float[2]);
    filament::math::float3 max(max_bounds_float[0], max_bounds_float[1],
                               max_bounds_float[2]);

    Box aabb;
    aabb.set(min, max);
    if (aabb.isEmpty()) {
        min.x -= 0.1f;
        min.y -= 0.1f;
        min.z -= 0.1f;
        max.x += 0.1f;
        max.y += 0.1f;
        max.z += 0.1f;
        aabb.set(min, max);
    }
    return aabb;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
