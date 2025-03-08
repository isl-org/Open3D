// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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

TGaussianSplatBuffersBuilder::TGaussianSplatBuffersBuilder(
        const t::geometry::PointCloud& geometry)
    : geometry_(geometry) {
    // Make sure geometry is on CPU
    auto pts = geometry.GetPointPositions();
    if (pts.IsCUDA()) {
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

RenderableManager::PrimitiveType TGaussianSplatBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::POINTS;
}

GeometryBuffersBuilder::Buffers TGaussianSplatBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const auto& points = geometry_.GetPointPositions();
    const size_t n_vertices = points.GetLength();

    // we usePOSITION for positions, COLOR for f_dc and opacity, TANGENTS for rot 
    // CUSTOM0 for scale, CUSTOM1-CUSTOM6 for f_rest. 
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(10)
                                 .vertexCount(uint32_t(n_vertices))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .attribute(VertexAttribute::COLOR, 1,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::TANGENTS, 2,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM0, 3,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM1, 4,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM2, 5,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM3, 6,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM4, 7,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM5, 8,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM6, 9,
                                    VertexBuffer::AttributeType::FLOAT4)
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

    const size_t color_array_size = n_vertices * 4 * sizeof(float);
    float* color_array = static_cast<float*>(malloc(color_array_size));
    if (geometry_.HasPointAttr("f_dc") && geometry_.HasPointAttr("opacity")) {
        float* f_dc_ptr = static_cast<float*>(geometry_.GetPointAttr("f_dc").GetDataPtr());
        float* opacity_ptr = static_cast<float*>(geometry_.GetPointAttr("opacity").GetDataPtr());
        for (size_t i = 0; i < n_vertices; i++) {
            std::memcpy(color_array + i * 4, f_dc_ptr + i * 3, 3 * sizeof(float));
            std::memcpy(color_array + i * 4 + 3, opacity_ptr + i, sizeof(float));
        }
    } else {
        for (size_t i = 0; i < n_vertices * 4; i++) {
            color_array[i] = 1.f;
        }
    }
    VertexBuffer::BufferDescriptor color_descriptor(
                color_array, color_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 1, std::move(color_descriptor));

    const size_t rot_array_size = n_vertices * 4 * sizeof(float);
    float* rot_array = static_cast<float*>(malloc(rot_array_size));
    if (geometry_.HasPointAttr("rot")) {
        memcpy(rot_array, geometry_.GetPointAttr("rot").GetDataPtr(),
               rot_array_size);
    } else {
        for (size_t i = 0; i < n_vertices * 4; ++i) {
            rot_array[i] = 1.f;
        }
    }
    VertexBuffer::BufferDescriptor rot_descriptor(
                rot_array, rot_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 2, std::move(rot_descriptor));

    const size_t scale_array_size = n_vertices * 4 * sizeof(float);
    float* scale_array = static_cast<float*>(malloc(scale_array_size));
    memset(scale_array, 0, scale_array_size);
    if (geometry_.HasPointAttr("scale")) {
        float* scale_src = static_cast<float*>(geometry_.GetPointAttr("scale").GetDataPtr());
        for (size_t i = 0; i < n_vertices; i++) {
            std::memcpy(scale_array + i * 4, scale_src + i * 3, 3*sizeof(float));
        }
    }
    VertexBuffer::BufferDescriptor scale_descriptor(
            scale_array, scale_array_size, GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 3, std::move(scale_descriptor));

    const size_t f_rest_array_size = n_vertices * 4 * sizeof(float);
    const size_t custom_buffer_num = 6;
    const size_t custom_buffer_start_index = 4;
    for(size_t i = 0; i < custom_buffer_num; i++) {
        float* f_rest_array = static_cast<float*>(malloc(f_rest_array_size));
        if (geometry_.HasPointAttr("f_rest")) {
            float* f_rest_src = static_cast<float*>(geometry_.GetPointAttr("f_rest").GetDataPtr());
            memcpy(f_rest_array, f_rest_src + i * n_vertices * 4, f_rest_array_size);
        } else {
            memset(f_rest_array, 0, f_rest_array_size);
        }
        VertexBuffer::BufferDescriptor f_rest_descriptor(
                    f_rest_array, f_rest_array_size, GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, custom_buffer_start_index + i, std::move(f_rest_descriptor));
    }

    auto ib_handle = CreateIndexBuffer(n_vertices);

    IndexBufferHandle downsampled_handle;
    if (n_vertices >= downsample_threshold_) {
        downsampled_handle =
                CreateIndexBuffer(n_vertices, downsample_threshold_);
    }

    return std::make_tuple(vb_handle, ib_handle, downsampled_handle);
}

filament::Box TGaussianSplatBuffersBuilder::ComputeAABB() {
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
