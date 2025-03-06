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

TGaussianBuffersBuilder::TGaussianBuffersBuilder(
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

RenderableManager::PrimitiveType TGaussianBuffersBuilder::GetPrimitiveType()
        const {
    return RenderableManager::PrimitiveType::POINTS;
}

GeometryBuffersBuilder::Buffers TGaussianBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const auto& points = geometry_.GetPointPositions();
    const size_t n_vertices = points.GetLength();

    // we use TANGENTS for normals, CUSTOM0 to load rot, CUSTOM1 to load scale, CUSTOM2 to load opacity, 
    // CUSTOM3 for f_dc, 
    VertexBuffer* vbuf = VertexBuffer::Builder()
                                 .bufferCount(8)
                                 .vertexCount(uint32_t(n_vertices))
                                 .attribute(VertexAttribute::POSITION, 0,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .normalized(VertexAttribute::COLOR)
                                 .attribute(VertexAttribute::COLOR, 1,
                                            VertexBuffer::AttributeType::FLOAT3)
                                 .normalized(VertexAttribute::TANGENTS)
                                 .attribute(VertexAttribute::TANGENTS, 2,
                                            VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM0, 3,
                                    VertexBuffer::AttributeType::FLOAT4)
                                 .attribute(VertexAttribute::CUSTOM1, 4,
                                    VertexBuffer::AttributeType::FLOAT3)
                                 .attribute(VertexAttribute::CUSTOM2, 5,
                                    VertexBuffer::AttributeType::FLOAT)
                                 .attribute(VertexAttribute::CUSTOM3, 6,
                                    VertexBuffer::AttributeType::FLOAT3)
                                // for now, we have not finish the defination of gaussian.mat file,
                                // so we allocate buffer for uv0 to avoid warning.
                                 .attribute(VertexAttribute::UV0, 7,
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
    // for now, we save f_dc in color buffer
    if (geometry_.HasPointAttr("f_dc")) {
        float* color_array = static_cast<float*>(malloc(color_array_size));
        memcpy(color_array, geometry_.GetPointAttr("f_dc").GetDataPtr(),
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

    const size_t rot_array_size = n_vertices * 4 * sizeof(float);
    if (geometry_.HasPointAttr("rot")) {
        float* rot_array = static_cast<float*>(malloc(rot_array_size));
        memcpy(rot_array, geometry_.GetPointAttr("rot").GetDataPtr(),
               rot_array_size);
        VertexBuffer::BufferDescriptor rot_descriptor(
                rot_array, rot_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 3, std::move(rot_descriptor));
    } else {
        float* rot_array = static_cast<float*>(malloc(rot_array_size));
        for (size_t i = 0; i < n_vertices * 4; ++i) {
            rot_array[i] = 1.f;
        }
        VertexBuffer::BufferDescriptor rot_descriptor(
                rot_array, rot_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, 3, std::move(rot_descriptor));
    }

    const size_t scale_array_size = n_vertices * 3 * sizeof(float);
    float* scale_array = static_cast<float*>(malloc(scale_array_size));
    if (geometry_.HasPointAttr("scale")) {
        const float* scale_src = static_cast<const float*>(
                geometry_.GetPointAttr("scale").GetDataPtr());
        memcpy(scale_array, scale_src, scale_array_size);
    } else {
        memset(scale_array, 0, scale_array_size);
    }
    VertexBuffer::BufferDescriptor scale_descriptor(
            scale_array, scale_array_size, GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 4, std::move(scale_descriptor));
    
    const size_t opacity_array_size = n_vertices * 1 * sizeof(float);
    float* opacity_array = static_cast<float*>(malloc(opacity_array_size));
    if (geometry_.HasPointAttr("opacity")) {
        const float* opacity_src = static_cast<const float*>(
                geometry_.GetPointAttr("opacity").GetDataPtr());
        memcpy(opacity_array, opacity_src, opacity_array_size);
    } else {
        memset(opacity_array, 0, opacity_array_size);
    }
    VertexBuffer::BufferDescriptor opacity_descriptor(
            opacity_array, opacity_array_size, GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 5, std::move(opacity_descriptor));

    const size_t f_dc_array_size = n_vertices * 3 * sizeof(float);
    float* f_dc_array = static_cast<float*>(malloc(f_dc_array_size));
    if (geometry_.HasPointAttr("f_dc")) {
        const float* f_dc_src = static_cast<const float*>(
                geometry_.GetPointAttr("f_dc").GetDataPtr());
        memcpy(f_dc_array, f_dc_src, f_dc_array_size);
    } else {
        memset(f_dc_array, 0, f_dc_array_size);
    }
    VertexBuffer::BufferDescriptor f_dc_descriptor(
            f_dc_array, f_dc_array_size, GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 6, std::move(f_dc_descriptor));

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
    vbuf->setBufferAt(engine, 7, std::move(uv_descriptor));

    auto ib_handle = CreateIndexBuffer(n_vertices);

    IndexBufferHandle downsampled_handle;
    if (n_vertices >= downsample_threshold_) {
        downsampled_handle =
                CreateIndexBuffer(n_vertices, downsample_threshold_);
    }

    return std::make_tuple(vb_handle, ib_handle, downsampled_handle);
}

filament::Box TGaussianBuffersBuilder::ComputeAABB() {
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
