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
    std::vector<std::string> check_list = {"f_dc", "opacity", "rot", "scale",
                                           "f_rest"};
    for (const auto& check_item : check_list) {
        if (check_item == "f_rest" && !geometry_.HasPointAttr(check_item)) {
            continue;
        }
        if (geometry_.GetPointAttr(check_item).GetDtype() != core::Float32) {
            auto check_item_instance = geometry_.GetPointAttr(check_item);
            utility::LogWarning(
                    "Tensor gaussian splat {} must have DType of Float32 not "
                    "{}. "
                    "Converting.",
                    check_item, check_item_instance.GetDtype().ToString());
            geometry_.GetPointAttr(check_item) =
                    check_item_instance.To(core::Float32);
        }
    }
}

GeometryBuffersBuilder::Buffers
TGaussianSplatBuffersBuilder::ConstructBuffers() {
    auto& engine = EngineInstance::GetInstance();
    auto& resource_mgr = EngineInstance::GetResourceManager();

    const auto& points = geometry_.GetPointPositions();
    const size_t n_vertices = points.GetLength();

    int sh_degree = geometry_.GaussianSplatGetSHOrder();
    if (sh_degree > 2) {
        utility::LogWarning(
                "Rendering for Gaussian splats with SH degrees higher than 2 "
                "is not supported. They are processed as SH degree 2.");
        sh_degree = 2;
    }

    int f_rest_coeffs_count = sh_degree * (sh_degree + 2) * 3;
    int f_rest_buffer_count = (f_rest_coeffs_count % 4 == 0)
                                      ? (f_rest_coeffs_count / 4)
                                      : std::ceil(f_rest_coeffs_count / 4.0);

    int base_buffer_count = 5;
    int all_buffer_count = base_buffer_count + f_rest_buffer_count;

    // we use POSITION for positions, COLOR for scale, CUSTOM7 for rot
    // CUSTOM0 for f_dc and opacity, CUSTOM1-CUSTOM6 for f_rest.
    VertexBuffer::Builder buffer_builder =
            VertexBuffer::Builder()
                    .bufferCount(all_buffer_count)
                    .vertexCount(uint32_t(n_vertices))
                    .attribute(VertexAttribute::POSITION, 0,
                               VertexBuffer::AttributeType::FLOAT3)
                    .attribute(VertexAttribute::COLOR, 1,
                               VertexBuffer::AttributeType::FLOAT4)
                    .attribute(VertexAttribute::TANGENTS, 2,
                               VertexBuffer::AttributeType::FLOAT)
                    .attribute(VertexAttribute::CUSTOM0, 3,
                               VertexBuffer::AttributeType::FLOAT4)
                    .attribute(VertexAttribute::CUSTOM7, 4,
                               VertexBuffer::AttributeType::FLOAT4);
    if (sh_degree >= 1) {
        buffer_builder.attribute(VertexAttribute::CUSTOM1, 5,
                                 VertexBuffer::AttributeType::FLOAT4);
        buffer_builder.attribute(VertexAttribute::CUSTOM2, 6,
                                 VertexBuffer::AttributeType::FLOAT4);
        buffer_builder.attribute(VertexAttribute::CUSTOM3, 7,
                                 VertexBuffer::AttributeType::FLOAT4);
    }
    if (sh_degree == 2) {
        buffer_builder.attribute(VertexAttribute::CUSTOM4, 8,
                                 VertexBuffer::AttributeType::FLOAT4);
        buffer_builder.attribute(VertexAttribute::CUSTOM5, 9,
                                 VertexBuffer::AttributeType::FLOAT4);
        buffer_builder.attribute(VertexAttribute::CUSTOM6, 10,
                                 VertexBuffer::AttributeType::FLOAT4);
    }

    VertexBuffer* vbuf = buffer_builder.build(engine);

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

    const size_t scale_array_size = n_vertices * 4 * sizeof(float);
    float* scale_array = static_cast<float*>(malloc(scale_array_size));
    std::memset(scale_array, 0, scale_array_size);
    float* scale_src = geometry_.GetPointAttr("scale").GetDataPtr<float>();
    for (size_t i = 0; i < n_vertices; i++) {
        std::memcpy(scale_array + i * 4, scale_src + i * 3, 3 * sizeof(float));
    }
    VertexBuffer::BufferDescriptor scale_descriptor(
            scale_array, scale_array_size,
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 1, std::move(scale_descriptor));

    // We need to allocate a buffer for TANGENTS; otherwise, Filament will issue
    // a warning.
    const size_t empty_array_size = n_vertices * sizeof(float);
    float* empty_array = static_cast<float*>(malloc(empty_array_size));
    std::memset(empty_array, 0, empty_array_size);
    VertexBuffer::BufferDescriptor empty_descriptor(
            empty_array, empty_array_size,
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 2, std::move(empty_descriptor));

    const size_t color_array_size = n_vertices * 4 * sizeof(float);
    float* color_array = static_cast<float*>(malloc(color_array_size));
    float* f_dc_ptr = geometry_.GetPointAttr("f_dc").GetDataPtr<float>();
    float* opacity_ptr = geometry_.GetPointAttr("opacity").GetDataPtr<float>();
    for (size_t i = 0; i < n_vertices; i++) {
        std::memcpy(color_array + i * 4, f_dc_ptr + i * 3, 3 * sizeof(float));
        std::memcpy(color_array + i * 4 + 3, opacity_ptr + i, sizeof(float));
    }
    VertexBuffer::BufferDescriptor color_descriptor(
            color_array, color_array_size,
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 3, std::move(color_descriptor));

    const size_t rot_array_size = n_vertices * 4 * sizeof(float);
    float* rot_array = static_cast<float*>(malloc(rot_array_size));
    std::memcpy(rot_array, geometry_.GetPointAttr("rot").GetDataPtr(),
                rot_array_size);
    VertexBuffer::BufferDescriptor rot_descriptor(
            rot_array, rot_array_size,
            GeometryBuffersBuilder::DeallocateBuffer);
    vbuf->setBufferAt(engine, 4, std::move(rot_descriptor));

    int data_count_in_one_buffer = 4;
    const size_t f_rest_array_size =
            n_vertices * data_count_in_one_buffer * sizeof(float);
    const size_t custom_buffer_start_index = 5;
    float* f_rest_src =
            (f_rest_buffer_count > 0)
                    ? geometry_.GetPointAttr("f_rest").GetDataPtr<float>()
                    : nullptr;
    for (int i = 0; i < f_rest_buffer_count; i++) {
        float* f_rest_array = static_cast<float*>(malloc(f_rest_array_size));

        size_t copy_data_size = f_rest_array_size;
        if (i == f_rest_buffer_count - 1) {
            int remaining_count_in_last_iter =
                    data_count_in_one_buffer +
                    (f_rest_coeffs_count -
                     f_rest_buffer_count * data_count_in_one_buffer);
            copy_data_size =
                    n_vertices * remaining_count_in_last_iter * sizeof(float);
            std::memset(f_rest_array, 0, f_rest_array_size);
        }

        std::memcpy(f_rest_array,
                    f_rest_src + i * n_vertices * data_count_in_one_buffer,
                    copy_data_size);
        VertexBuffer::BufferDescriptor f_rest_descriptor(
                f_rest_array, f_rest_array_size,
                GeometryBuffersBuilder::DeallocateBuffer);
        vbuf->setBufferAt(engine, custom_buffer_start_index + i,
                          std::move(f_rest_descriptor));
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
