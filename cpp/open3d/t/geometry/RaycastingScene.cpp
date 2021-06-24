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

#include "open3d/t/geometry/RaycastingScene.h"

#include <embree3/rtcore.h>

#include <tuple>
#include <vector>

#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace {

void error_function(void* userPtr, enum RTCError error, const char* str) {
    open3d::utility::LogError("embree error: {} {}", error, str);
}

}  // namespace

namespace open3d {
namespace t {
namespace geometry {

struct RaycastingScene::Impl {
    RTCDevice device;
    RTCScene scene;
    bool scene_committed;  // true if the scene has been committed
    // vector for storing some information about the added geometry
    std::vector<std::tuple<RTCGeometryType, const void*, const void*>>
            geometry_ptrs;
    core::Device tensor_device;  // cpu

    template <bool LINE_INTERSECTION>
    void CastRays(const float* const rays,
                  const size_t num_rays,
                  float* t_hit,
                  unsigned int* geometry_ids,
                  unsigned int* primitive_ids,
                  float* primitive_uvs,
                  float* primitive_normals) {
        if (!scene_committed) {
            rtcCommitScene(scene);
            scene_committed = true;
        }

        struct RTCIntersectContext context;
        rtcInitIntersectContext(&context);

        const size_t max_batch_size = 1048576;

        std::vector<RTCRayHit> rayhits(std::min(num_rays, max_batch_size));

        const int num_batches = utility::DivUp(num_rays, rayhits.size());

        for (int n = 0; n < num_batches; ++n) {
            size_t start_idx = n * rayhits.size();
            size_t end_idx = std::min(num_rays, (n + 1) * rayhits.size());

            for (size_t i = start_idx; i < end_idx; ++i) {
                RTCRayHit& rh = rayhits[i - start_idx];
                const float* r = &rays[i * 6];
                rh.ray.org_x = r[0];
                rh.ray.org_y = r[1];
                rh.ray.org_z = r[2];
                if (LINE_INTERSECTION) {
                    rh.ray.dir_x = r[3] - r[0];
                    rh.ray.dir_y = r[4] - r[1];
                    rh.ray.dir_z = r[5] - r[2];
                } else {
                    rh.ray.dir_x = r[3];
                    rh.ray.dir_y = r[4];
                    rh.ray.dir_z = r[5];
                }
                rh.ray.tnear = 0;
                if (LINE_INTERSECTION) {
                    rh.ray.tfar = 1.f;
                } else {
                    rh.ray.tfar = std::numeric_limits<float>::infinity();
                }
                rh.ray.mask = 0;
                rh.ray.id = i - start_idx;
                rh.ray.flags = 0;
                rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            }

            rtcIntersect1M(scene, &context, &rayhits[0], end_idx - start_idx,
                           sizeof(RTCRayHit));

            for (size_t i = start_idx; i < end_idx; ++i) {
                RTCRayHit rh = rayhits[i - start_idx];
                size_t idx = rh.ray.id + start_idx;
                t_hit[idx] = rh.ray.tfar;
                if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                    geometry_ids[idx] = rh.hit.geomID;
                    primitive_ids[idx] = rh.hit.primID;
                    primitive_uvs[idx * 2 + 0] = rh.hit.u;
                    primitive_uvs[idx * 2 + 1] = rh.hit.v;
                    float inv_norm = 1.f / std::sqrt(rh.hit.Ng_x * rh.hit.Ng_x +
                                                     rh.hit.Ng_y * rh.hit.Ng_y +
                                                     rh.hit.Ng_z * rh.hit.Ng_z);
                    primitive_normals[idx * 3 + 0] = rh.hit.Ng_x * inv_norm;
                    primitive_normals[idx * 3 + 1] = rh.hit.Ng_y * inv_norm;
                    primitive_normals[idx * 3 + 2] = rh.hit.Ng_z * inv_norm;
                } else {
                    geometry_ids[idx] = RTC_INVALID_GEOMETRY_ID;
                    primitive_ids[idx] = RTC_INVALID_GEOMETRY_ID;
                    primitive_uvs[idx * 2 + 0] = 0;
                    primitive_uvs[idx * 2 + 1] = 0;
                    primitive_normals[idx * 3 + 0] = 0;
                    primitive_normals[idx * 3 + 1] = 0;
                    primitive_normals[idx * 3 + 2] = 0;
                }
            }
        }
    }
};

RaycastingScene::RaycastingScene() : impl_(new RaycastingScene::Impl()) {
    impl_->device = rtcNewDevice(NULL);
    rtcSetDeviceErrorFunction(impl_->device, error_function, NULL);

    impl_->scene = rtcNewScene(impl_->device);
    // set flag for better accuracy
    rtcSetSceneFlags(
            impl_->scene,
            RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);

    impl_->scene_committed = false;
}

RaycastingScene::~RaycastingScene() {
    rtcReleaseScene(impl_->scene);
    rtcReleaseDevice(impl_->device);
}

uint32_t RaycastingScene::AddTriangles(const core::Tensor& vertices,
                                       const core::Tensor& triangles) {
    vertices.AssertDevice(impl_->tensor_device);
    vertices.AssertShapeCompatible({utility::nullopt, 3});
    vertices.AssertDtype(core::Dtype::FromType<float>());
    triangles.AssertDevice(impl_->tensor_device);
    triangles.AssertShapeCompatible({utility::nullopt, 3});
    triangles.AssertDtype(core::Dtype::FromType<uint32_t>());

    const size_t num_vertices = vertices.GetLength();
    const size_t num_triangles = triangles.GetLength();

    // scene needs to be recommitted
    impl_->scene_committed = false;
    RTCGeometry geom =
            rtcNewGeometry(impl_->device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // rtcSetNewGeometryBuffer will take care of alignment and padding
    float* vertex_buffer = (float*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            3 * sizeof(float), num_vertices);

    uint32_t* index_buffer = (uint32_t*)rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            3 * sizeof(uint32_t), num_triangles);

    {
        auto data = vertices.Contiguous();
        memcpy(vertex_buffer, data.GetDataPtr(),
               sizeof(float) * 3 * num_vertices);
    }
    {
        auto data = triangles.Contiguous();
        memcpy(index_buffer, data.GetDataPtr(),
               sizeof(uint32_t) * 3 * num_triangles);
    }
    rtcCommitGeometry(geom);

    uint32_t geom_id = rtcAttachGeometry(impl_->scene, geom);
    rtcReleaseGeometry(geom);

    impl_->geometry_ptrs.push_back(std::make_tuple(RTC_GEOMETRY_TYPE_TRIANGLE,
                                                   (const void*)vertex_buffer,
                                                   (const void*)index_buffer));
    return geom_id;
}

std::unordered_map<std::string, core::Tensor> RaycastingScene::CastRays(
        const core::Tensor& rays) {
    rays.AssertDevice(impl_->tensor_device);
    rays.AssertShapeCompatible({utility::nullopt, 6});
    if (rays.GetShape().size() < 2) {
        utility::LogError("rays Tensor ndim is {} but expected ndim >= 2",
                          rays.GetShape().size());
    }
    if (rays.GetShape().back() != 6) {
        utility::LogError(
                "The last dimension of the rays Tensor must be 6 but got "
                "Tensor with shape {}",
                rays.GetShape().ToString());
    }
    rays.AssertDtype(core::Dtype::FromType<float>());

    auto shape = rays.GetShape();
    shape.pop_back();
    size_t num_rays = 1;
    for (auto s : shape) {
        num_rays *= s;
    }

    std::unordered_map<std::string, core::Tensor> result;
    result["t_hit"] = core::Tensor(shape, core::Dtype::FromType<float>());
    result["geometry_ids"] =
            core::Tensor(shape, core::Dtype::FromType<uint32_t>());
    result["primitive_ids"] =
            core::Tensor(shape, core::Dtype::FromType<uint32_t>());
    shape.push_back(2);
    result["primitive_uvs"] =
            core::Tensor(shape, core::Dtype::FromType<float>());
    shape.back() = 3;
    result["normals"] = core::Tensor(shape, core::Dtype::FromType<float>());

    auto data = rays.Contiguous();
    impl_->CastRays<false>(data.GetDataPtr<float>(), num_rays,
                           result["t_hit"].GetDataPtr<float>(),
                           result["geometry_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_uvs"].GetDataPtr<float>(),
                           result["normals"].GetDataPtr<float>());

    return result;
}

const uint32_t RaycastingScene::INVALID_ID = RTC_INVALID_GEOMETRY_ID;

}  // namespace geometry
}  // namespace t
}  // namespace open3d
