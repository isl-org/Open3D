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
TGaussianSplatBuffersBuilder::TGaussianSplatBuffersBuilder(
        const t::geometry::PointCloud& geometry)
    : TPointCloudBuffersBuilder(geometry) {
    if (!geometry.IsGaussianSplat()) {
        utility::LogWarning(
            "TGaussianSplatBuffers is constructed for a geometry that is not GaussianSplat."
        );
    }

    std::vector<std::string> check_list = {"f_dc", "opacity", "rot", "scale", "f_rest"};
    for (const auto& check_item : check_list) {
        if (geometry_.HasPointAttr(check_item) &&
            geometry_.GetPointAttr(check_item).GetDtype() != core::Float32) {
            auto check_item_instance = geometry_.GetPointAttr(check_item);
            utility::LogWarning(
                    "Tensor gaussian splat {} must have DType of Float32 not {}. "
                    "Converting.",
                    check_item, check_item_instance.GetDtype().ToString());
            geometry_.GetPointAttr(check_item) = check_item_instance.To(core::Float32);
        }
    }
}

GeometryBuffersBuilder::Buffers TGaussianSplatBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const auto& points = geometry_.GetPointPositions();
    const size_t n_vertices = points.GetLength();

    // we use POSITION for positions, COLOR for f_dc and opacity, TANGENTS for rot 
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
        std::memcpy(rot_array, geometry_.GetPointAttr("rot").GetDataPtr(),
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
    std::memset(scale_array, 0, scale_array_size);
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
            std::memcpy(f_rest_array, f_rest_src + i * n_vertices * 4, f_rest_array_size);
        } else {
            std::memset(f_rest_array, 0, f_rest_array_size);
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
}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
