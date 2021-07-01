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

#ifdef _MSC_VER
// embree header files in tutorials/common redefine some macros on win
#pragma warning(disable : 4005)
#endif
#include "open3d/t/geometry/RaycastingScene.h"

// This header is in the embree src dir (embree/src/ext_embree/..).
#include <embree3/rtcore.h>
#include <tutorials/common/math/closest_point.h>

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

// The maximum number of rays used in calls to embree.
static const size_t MAX_BATCH_SIZE = 1048576;

namespace {

// Error function called by embree.
void ErrorFunction(void* userPtr, enum RTCError error, const char* str) {
    open3d::utility::LogError("embree error: {} {}", error, str);
}

// Checks the last dim, ensures that the number of dims is >= min_ndim, checks
// the device, and dtype.
template <class DTYPE>
void AssertTensorDtypeLastDimDeviceMinNDim(const open3d::core::Tensor& tensor,
                                           const std::string& tensor_name,
                                           int64_t last_dim,
                                           const open3d::core::Device& device,
                                           int64_t min_ndim = 2) {
    tensor.AssertDevice(device);
    if (tensor.NumDims() < min_ndim) {
        open3d::utility::LogError(
                "{} Tensor ndim is {} but expected ndim >= {}", tensor_name,
                tensor.NumDims(), min_ndim);
    }
    if (tensor.GetShape().back() != last_dim) {
        open3d::utility::LogError(
                "The last dimension of the {} Tensor must be {} but got "
                "Tensor with shape {}",
                tensor_name, last_dim, tensor.GetShape().ToString());
    }
    tensor.AssertDtype(open3d::core::Dtype::FromType<DTYPE>());
}

struct CountIntersectionsContext {
    RTCIntersectContext context;
    std::vector<std::tuple<uint32_t, uint32_t, float>>*
            previous_geom_prim_ID_tfar;
    int* intersections;
};

void CountIntersectionsFunc(const RTCFilterFunctionNArguments* args) {
    int* valid = args->valid;
    const CountIntersectionsContext* context =
            reinterpret_cast<const CountIntersectionsContext*>(args->context);
    struct RTCRayN* rayN = args->ray;
    struct RTCHitN* hitN = args->hit;
    const unsigned int N = args->N;

    // Avoid crashing when debug visualizations are used.
    if (context == nullptr) return;

    std::vector<std::tuple<uint32_t, uint32_t, float>>*
            previous_geom_prim_ID_tfar = context->previous_geom_prim_ID_tfar;
    int* intersections = context->intersections;

    // Iterate over all rays in ray packet.
    for (unsigned int ui = 0; ui < N; ui += 1) {
        // Calculate loop and execution mask
        unsigned int vi = ui + 0;
        if (vi >= N) continue;

        // Ignore inactive rays.
        if (valid[vi] != -1) continue;

        // Read ray/hit from ray structure.
        RTCRay ray = rtcGetRayFromRayN(rayN, N, ui);
        RTCHit hit = rtcGetHitFromHitN(hitN, N, ui);

        unsigned int ray_id = ray.id;
        std::tuple<uint32_t, uint32_t, float> gpID(hit.geomID, hit.primID,
                                                   ray.tfar);
        auto& prev_gpIDtfar = previous_geom_prim_ID_tfar->operator[](ray_id);
        if (std::get<0>(prev_gpIDtfar) != hit.geomID ||
            (std::get<1>(prev_gpIDtfar) != hit.primID &&
             std::get<2>(prev_gpIDtfar) != ray.tfar)) {
            ++(intersections[ray_id]);
            previous_geom_prim_ID_tfar->operator[](ray_id) = gpID;
        }
        // Always ignore hit
        valid[ui] = 0;
    }
}

struct ClosestPointResult {
    ClosestPointResult()
        : primID(RTC_INVALID_GEOMETRY_ID),
          geomID(RTC_INVALID_GEOMETRY_ID),
          geometry_ptrs_ptr() {}

    embree::Vec3f p;
    unsigned int primID;
    unsigned int geomID;
    std::vector<std::tuple<RTCGeometryType, const void*, const void*>>*
            geometry_ptrs_ptr;
};

// Code adapted from the embree closest_point tutorial.
bool ClosestPointFunc(RTCPointQueryFunctionArguments* args) {
    using namespace embree;
    assert(args->userPtr);
    const unsigned int geomID = args->geomID;
    const unsigned int primID = args->primID;

    // query position in world space
    Vec3fa q(args->query->x, args->query->y, args->query->z);

    ClosestPointResult* result =
            static_cast<ClosestPointResult*>(args->userPtr);
    const RTCGeometryType geom_type =
            std::get<0>(result->geometry_ptrs_ptr->operator[](geomID));
    const void* ptr1 =
            std::get<1>(result->geometry_ptrs_ptr->operator[](geomID));
    const void* ptr2 =
            std::get<2>(result->geometry_ptrs_ptr->operator[](geomID));

    if (RTC_GEOMETRY_TYPE_TRIANGLE == geom_type) {
        const float* vertices = (const float*)ptr1;
        const uint32_t* triangles = (const uint32_t*)ptr2;

        Vec3fa v0(vertices[3 * triangles[3 * primID + 0] + 0],
                  vertices[3 * triangles[3 * primID + 0] + 1],
                  vertices[3 * triangles[3 * primID + 0] + 2]);
        Vec3fa v1(vertices[3 * triangles[3 * primID + 1] + 0],
                  vertices[3 * triangles[3 * primID + 1] + 1],
                  vertices[3 * triangles[3 * primID + 1] + 2]);
        Vec3fa v2(vertices[3 * triangles[3 * primID + 2] + 0],
                  vertices[3 * triangles[3 * primID + 2] + 1],
                  vertices[3 * triangles[3 * primID + 2] + 2]);

        // Determine distance to closest point on triangle (implemented in
        // common/math/closest_point.h).
        const Vec3fa p = closestPointTriangle(q, v0, v1, v2);
        float d = distance(q, p);

        // Store result in userPtr and update the query radius if we found a
        // point closer to the query position. This is optional but allows for
        // faster traversal (due to better culling).
        if (d < args->query->radius) {
            args->query->radius = d;
            result->p = p;
            result->primID = primID;
            result->geomID = geomID;
            return true;  // Return true to indicate that the query radius
                          // changed.
        }
    }
    return false;
}

}  // namespace

namespace open3d {
namespace t {
namespace geometry {

struct RaycastingScene::Impl {
    RTCDevice device_;
    RTCScene scene_;
    bool scene_committed_;  // true if the scene has been committed.
    // Vector for storing some information about the added geometry.
    std::vector<std::tuple<RTCGeometryType, const void*, const void*>>
            geometry_ptrs_;
    core::Device tensor_device_;  // cpu

    template <bool LINE_INTERSECTION>
    void CastRays(const float* const rays,
                  const size_t num_rays,
                  float* t_hit,
                  unsigned int* geometry_ids,
                  unsigned int* primitive_ids,
                  float* primitive_uvs,
                  float* primitive_normals) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        struct RTCIntersectContext context;
        rtcInitIntersectContext(&context);

        std::vector<RTCRayHit> rayhits(std::min(num_rays, MAX_BATCH_SIZE));

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

            rtcIntersect1M(scene_, &context, &rayhits[0], end_idx - start_idx,
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

    void CountIntersections(const float* const rays,
                            const size_t num_rays,
                            int* intersections) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        memset(intersections, 0, sizeof(int) * num_rays);

        std::vector<std::tuple<uint32_t, uint32_t, float>>
                previous_geom_prim_ID_tfar(
                        num_rays,
                        std::make_tuple(uint32_t(RTC_INVALID_GEOMETRY_ID),
                                        uint32_t(RTC_INVALID_GEOMETRY_ID),
                                        0.f));

        CountIntersectionsContext context;
        rtcInitIntersectContext(&context.context);
        context.context.filter = CountIntersectionsFunc;
        context.previous_geom_prim_ID_tfar = &previous_geom_prim_ID_tfar;
        context.intersections = intersections;

        std::vector<RTCRayHit> rayhits(std::min(num_rays, MAX_BATCH_SIZE));

        const int num_batches = utility::DivUp(num_rays, rayhits.size());

        for (int n = 0; n < num_batches; ++n) {
            size_t start_idx = n * rayhits.size();
            size_t end_idx = std::min(num_rays, (n + 1) * rayhits.size());

            for (size_t i = start_idx; i < end_idx; ++i) {
                RTCRayHit* rh = &rayhits[i - start_idx];
                const float* r = &rays[i * 6];
                rh->ray.org_x = r[0];
                rh->ray.org_y = r[1];
                rh->ray.org_z = r[2];
                rh->ray.dir_x = r[3];
                rh->ray.dir_y = r[4];
                rh->ray.dir_z = r[5];
                rh->ray.tnear = 0;
                rh->ray.tfar = std::numeric_limits<float>::infinity();
                rh->ray.mask = 0;
                rh->ray.flags = 0;
                rh->ray.id = i;
                rh->hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rh->hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            }

            rtcIntersect1M(scene_, &context.context, &rayhits[0],
                           end_idx - start_idx, sizeof(RTCRayHit));
        }
    }

    void ComputeClosestPoints(const float* const query_points,
                              const size_t num_query_points,
                              float* closest_points,
                              unsigned int* geometry_ids,
                              unsigned int* primitive_ids) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        for (size_t i = 0; i < num_query_points; ++i) {
            RTCPointQuery query;
            query.x = query_points[i * 3 + 0];
            query.y = query_points[i * 3 + 1];
            query.z = query_points[i * 3 + 2];
            query.radius = std::numeric_limits<float>::infinity();
            query.time = 0.f;

            ClosestPointResult result;
            result.geometry_ptrs_ptr = &geometry_ptrs_;

            RTCPointQueryContext instStack;
            rtcInitPointQueryContext(&instStack);
            rtcPointQuery(scene_, &query, &instStack, &ClosestPointFunc,
                          (void*)&result);

            closest_points[3 * i + 0] = result.p.x;
            closest_points[3 * i + 1] = result.p.y;
            closest_points[3 * i + 2] = result.p.z;
            geometry_ids[i] = result.geomID;
            primitive_ids[i] = result.primID;
        }
    }
};

RaycastingScene::RaycastingScene() : impl_(new RaycastingScene::Impl()) {
    impl_->device_ = rtcNewDevice(NULL);
    rtcSetDeviceErrorFunction(impl_->device_, ErrorFunction, NULL);

    impl_->scene_ = rtcNewScene(impl_->device_);
    // set flag for better accuracy
    rtcSetSceneFlags(
            impl_->scene_,
            RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);

    impl_->scene_committed_ = false;
}

RaycastingScene::~RaycastingScene() {
    rtcReleaseScene(impl_->scene_);
    rtcReleaseDevice(impl_->device_);
}

uint32_t RaycastingScene::AddTriangles(const core::Tensor& vertices,
                                       const core::Tensor& triangles) {
    vertices.AssertDevice(impl_->tensor_device_);
    vertices.AssertShapeCompatible({utility::nullopt, 3});
    vertices.AssertDtype(core::Dtype::Float32);
    triangles.AssertDevice(impl_->tensor_device_);
    triangles.AssertShapeCompatible({utility::nullopt, 3});
    triangles.AssertDtype(core::Dtype::UInt32);

    const size_t num_vertices = vertices.GetLength();
    const size_t num_triangles = triangles.GetLength();

    // scene needs to be recommitted
    impl_->scene_committed_ = false;
    RTCGeometry geom =
            rtcNewGeometry(impl_->device_, RTC_GEOMETRY_TYPE_TRIANGLE);

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

    uint32_t geom_id = rtcAttachGeometry(impl_->scene_, geom);
    rtcReleaseGeometry(geom);

    impl_->geometry_ptrs_.push_back(std::make_tuple(RTC_GEOMETRY_TYPE_TRIANGLE,
                                                    (const void*)vertex_buffer,
                                                    (const void*)index_buffer));
    return geom_id;
}

uint32_t RaycastingScene::AddTriangles(const TriangleMesh& mesh) {
    size_t num_verts = mesh.GetVertices().GetLength();
    if (num_verts > std::numeric_limits<uint32_t>::max()) {
        utility::LogError(
                "Cannot add mesh with more than {} vertices to the scene",
                std::numeric_limits<uint32_t>::max());
    }
    return AddTriangles(mesh.GetVertices(),
                        mesh.GetTriangles().To(core::Dtype::UInt32));
}

std::unordered_map<std::string, core::Tensor> RaycastingScene::CastRays(
        const core::Tensor& rays) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(rays, "rays", 6,
                                                 impl_->tensor_device_);
    auto shape = rays.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_rays = shape.NumElements();

    std::unordered_map<std::string, core::Tensor> result;
    result["t_hit"] = core::Tensor(shape, core::Dtype::Float32);
    result["geometry_ids"] = core::Tensor(shape, core::Dtype::UInt32);
    result["primitive_ids"] = core::Tensor(shape, core::Dtype::UInt32);
    shape.push_back(2);
    result["primitive_uvs"] = core::Tensor(shape, core::Dtype::Float32);
    shape.back() = 3;
    result["primitive_normals"] = core::Tensor(shape, core::Dtype::Float32);

    auto data = rays.Contiguous();
    impl_->CastRays<false>(data.GetDataPtr<float>(), num_rays,
                           result["t_hit"].GetDataPtr<float>(),
                           result["geometry_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_uvs"].GetDataPtr<float>(),
                           result["primitive_normals"].GetDataPtr<float>());

    return result;
}

core::Tensor RaycastingScene::CountIntersections(const core::Tensor& rays) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(rays, "rays", 6,
                                                 impl_->tensor_device_);
    auto shape = rays.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_rays = shape.NumElements();

    core::Tensor intersections(shape, core::Dtype::FromType<int>());

    auto data = rays.Contiguous();

    impl_->CountIntersections(data.GetDataPtr<float>(), num_rays,
                              intersections.GetDataPtr<int>());
    return intersections;
}

std::unordered_map<std::string, core::Tensor>
RaycastingScene::ComputeClosestPoints(const core::Tensor& query_points) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();

    std::unordered_map<std::string, core::Tensor> result;
    result["geometry_ids"] = core::Tensor(shape, core::Dtype::UInt32);
    result["primitive_ids"] = core::Tensor(shape, core::Dtype::UInt32);
    shape.push_back(3);
    result["points"] = core::Tensor(shape, core::Dtype::Float32);

    auto data = query_points.Contiguous();
    impl_->ComputeClosestPoints(data.GetDataPtr<float>(), num_query_points,
                                result["points"].GetDataPtr<float>(),
                                result["geometry_ids"].GetDataPtr<uint32_t>(),
                                result["primitive_ids"].GetDataPtr<uint32_t>());

    return result;
}

core::Tensor RaycastingScene::ComputeDistance(
        const core::Tensor& query_points) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.

    auto data = query_points.Contiguous();
    auto closest_points = ComputeClosestPoints(data);

    size_t num_query_points = shape.NumElements();
    Eigen::Map<Eigen::MatrixXf> query_points_map(data.GetDataPtr<float>(), 3,
                                                 num_query_points);
    Eigen::Map<Eigen::MatrixXf> closest_points_map(
            closest_points["points"].GetDataPtr<float>(), 3, num_query_points);
    core::Tensor distance(shape, core::Dtype::Float32);
    Eigen::Map<Eigen::VectorXf> distance_map(distance.GetDataPtr<float>(),
                                             num_query_points);

    distance_map = (closest_points_map - query_points_map).colwise().norm();
    return distance;
}

core::Tensor RaycastingScene::ComputeSignedDistance(
        const core::Tensor& query_points) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();

    auto data = query_points.Contiguous();
    auto distance = ComputeDistance(data);
    core::Tensor rays({int64_t(num_query_points), 6}, core::Dtype::Float32);
    rays.SetItem({core::TensorKey::Slice(0, num_query_points, 1),
                  core::TensorKey::Slice(0, 3, 1)},
                 data.Reshape({int64_t(num_query_points), 3}));
    rays.SetItem(
            {core::TensorKey::Slice(0, num_query_points, 1),
             core::TensorKey::Slice(3, 6, 1)},
            core::Tensor::Ones({1}, core::Dtype::Float32, impl_->tensor_device_)
                    .Expand({int64_t(num_query_points), 3}));
    auto intersections = CountIntersections(rays);

    Eigen::Map<Eigen::VectorXf> distance_map(distance.GetDataPtr<float>(),
                                             num_query_points);
    Eigen::Map<Eigen::VectorXi> intersections_map(
            intersections.GetDataPtr<int>(), num_query_points);
    intersections_map = intersections_map.unaryExpr(
            [](const int x) { return (x % 2) ? -1 : 1; });
    distance_map.array() *= intersections_map.array().cast<float>();
    return distance;
}

core::Tensor RaycastingScene::ComputeOccupancy(
        const core::Tensor& query_points) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();

    core::Tensor rays({int64_t(num_query_points), 6}, core::Dtype::Float32);
    rays.SetItem({core::TensorKey::Slice(0, num_query_points, 1),
                  core::TensorKey::Slice(0, 3, 1)},
                 query_points.Reshape({int64_t(num_query_points), 3}));
    rays.SetItem(
            {core::TensorKey::Slice(0, num_query_points, 1),
             core::TensorKey::Slice(3, 6, 1)},
            core::Tensor::Ones({1}, core::Dtype::Float32, impl_->tensor_device_)
                    .Expand({int64_t(num_query_points), 3}));
    auto intersections = CountIntersections(rays);
    Eigen::Map<Eigen::VectorXi> intersections_map(
            intersections.GetDataPtr<int>(), num_query_points);
    intersections_map =
            intersections_map.unaryExpr([](const int x) { return x % 2; });
    return intersections.To(core::Dtype::Float32).Reshape(shape);
}

core::Tensor RaycastingScene::CreateRaysPinhole(
        const core::Tensor& intrinsic_matrix,
        const core::Tensor& extrinsic_matrix,
        int width_px,
        int height_px) {
    intrinsic_matrix.AssertDevice(core::Device());
    intrinsic_matrix.AssertShape({3, 3});
    extrinsic_matrix.AssertDevice(core::Device());
    extrinsic_matrix.AssertShape({4, 4});

    auto intrinsic_matrix_contig =
            intrinsic_matrix.To(core::Dtype::Float64).Contiguous();
    auto extrinsic_matrix_contig =
            extrinsic_matrix.To(core::Dtype::Float64).Contiguous();
    // Eigen is col major
    Eigen::Map<Eigen::MatrixXd> KT(intrinsic_matrix_contig.GetDataPtr<double>(),
                                   3, 3);
    Eigen::Map<Eigen::MatrixXd> TT(extrinsic_matrix_contig.GetDataPtr<double>(),
                                   4, 4);

    Eigen::Matrix3d invK = KT.transpose().inverse();
    Eigen::Matrix3d RT = TT.block(0, 0, 3, 3);
    Eigen::Vector3d t = TT.transpose().block(0, 3, 3, 1);
    Eigen::Vector3d C = -RT * t;
    Eigen::Matrix3f RT_invK = (RT * invK).cast<float>();

    core::Tensor rays({height_px, width_px, 6}, core::Dtype::Float32);
    Eigen::Map<Eigen::MatrixXf> rays_map(rays.GetDataPtr<float>(), 6,
                                         height_px * width_px);

    Eigen::Matrix<float, 6, 1> r;
    r.topRows<3>() = C.cast<float>();
    int64_t linear_idx = 0;
    for (int y = 0; y < height_px; ++y) {
        for (int x = 0; x < width_px; ++x, ++linear_idx) {
            Eigen::Vector3f px(x + 0.5f, y + 0.5f, 1);
            Eigen::Vector3f ray_dir = RT_invK * px;
            r.bottomRows<3>() = ray_dir;
            rays_map.col(linear_idx) = r;
        }
    }
    return rays;
}

core::Tensor RaycastingScene::CreateRaysPinhole(double fov_deg,
                                                const core::Tensor& center,
                                                const core::Tensor& eye,
                                                const core::Tensor& up,
                                                int width_px,
                                                int height_px) {
    center.AssertDevice(core::Device());
    center.AssertShape({3});
    eye.AssertDevice(core::Device());
    eye.AssertShape({3});
    up.AssertDevice(core::Device());
    up.AssertShape({3});

    double focal_length =
            0.5 * width_px / std::tan(0.5 * (M_PI / 180) * fov_deg);

    core::Tensor intrinsic_matrix =
            core::Tensor::Eye(3, core::Dtype::Float64, core::Device());
    Eigen::Map<Eigen::MatrixXd> intrinsic_matrix_map(
            intrinsic_matrix.GetDataPtr<double>(), 3, 3);
    intrinsic_matrix_map(0, 0) = focal_length;
    intrinsic_matrix_map(1, 1) = focal_length;
    intrinsic_matrix_map(2, 0) = 0.5 * width_px;
    intrinsic_matrix_map(2, 1) = 0.5 * height_px;

    auto center_contig = center.To(core::Dtype::Float64).Contiguous();
    auto eye_contig = eye.To(core::Dtype::Float64).Contiguous();
    auto up_contig = up.To(core::Dtype::Float64).Contiguous();

    Eigen::Map<const Eigen::Vector3d> center_map(
            center_contig.GetDataPtr<double>());
    Eigen::Map<const Eigen::Vector3d> eye_map(eye_contig.GetDataPtr<double>());
    Eigen::Map<const Eigen::Vector3d> up_map(up_contig.GetDataPtr<double>());

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R.row(1) = up_map / up_map.norm();
    R.row(2) = center_map - eye_map;
    R.row(2) /= R.row(2).norm();
    R.row(0) = R.row(1).cross(R.row(2));
    R.row(0) /= R.row(0).norm();
    R.row(1) = R.row(2).cross(R.row(0));
    Eigen::Vector3d t = -R * eye_map;

    core::Tensor extrinsic_matrix =
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device());
    Eigen::Map<Eigen::MatrixXd> extrinsic_matrix_map(
            extrinsic_matrix.GetDataPtr<double>(), 4, 4);
    extrinsic_matrix_map.block(3, 0, 1, 3) = t.transpose();
    extrinsic_matrix_map.block(0, 0, 3, 3) = R.transpose();

    return CreateRaysPinhole(intrinsic_matrix, extrinsic_matrix, width_px,
                             height_px);
}

uint32_t RaycastingScene::INVALID_ID() { return RTC_INVALID_GEOMETRY_ID; }

}  // namespace geometry
}  // namespace t
}  // namespace open3d
