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
#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <tuple>
#include <unsupported/Eigen/AlignedVector3>
#include <vector>

#include "open3d/core/TensorCheck.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace {

typedef Eigen::AlignedVector3<float> Vec3fa;
// Dont force alignment for Vec2f because we use it just for storing
typedef Eigen::Matrix<float, 2, 1, Eigen::DontAlign> Vec2f;
typedef Eigen::Vector3f Vec3f;

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
    open3d::core::AssertTensorDevice(tensor, device);
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
    open3d::core::AssertTensorDtype(tensor,
                                    open3d::core::Dtype::FromType<DTYPE>());
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

// Adapted from common/math/closest_point.h
inline Vec3fa closestPointTriangle(Vec3fa const& p,
                                   Vec3fa const& a,
                                   Vec3fa const& b,
                                   Vec3fa const& c,
                                   float& tex_u,
                                   float& tex_v) {
    const Vec3fa ab = b - a;
    const Vec3fa ac = c - a;
    const Vec3fa ap = p - a;

    const float d1 = ab.dot(ap);
    const float d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f) {
        tex_u = 0;
        tex_v = 0;
        return a;
    }

    const Vec3fa bp = p - b;
    const float d3 = ab.dot(bp);
    const float d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3) {
        tex_u = 1;
        tex_v = 0;
        return b;
    }

    const Vec3fa cp = p - c;
    const float d5 = ab.dot(cp);
    const float d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6) {
        tex_u = 0;
        tex_v = 1;
        return c;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        const float v = d1 / (d1 - d3);
        tex_u = v;
        tex_v = 0;
        return a + v * ab;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float v = d2 / (d2 - d6);
        tex_u = 0;
        tex_v = v;
        return a + v * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        tex_u = 1 - v;
        tex_v = v;
        return b + v * (c - b);
    }

    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    tex_u = v;
    tex_v = w;
    return a + v * ab + w * ac;
}

struct ClosestPointResult {
    ClosestPointResult()
        : primID(RTC_INVALID_GEOMETRY_ID),
          geomID(RTC_INVALID_GEOMETRY_ID),
          geometry_ptrs_ptr() {}

    Vec3f p;
    unsigned int primID;
    unsigned int geomID;
    Vec2f uv;
    Vec3f n;
    std::vector<std::tuple<RTCGeometryType, const void*, const void*>>*
            geometry_ptrs_ptr;
};

// Code adapted from the embree closest_point tutorial.
bool ClosestPointFunc(RTCPointQueryFunctionArguments* args) {
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
        const float* vertex_positions = (const float*)ptr1;
        const uint32_t* triangle_indices = (const uint32_t*)ptr2;

        Vec3fa v0(vertex_positions[3 * triangle_indices[3 * primID + 0] + 0],
                  vertex_positions[3 * triangle_indices[3 * primID + 0] + 1],
                  vertex_positions[3 * triangle_indices[3 * primID + 0] + 2]);
        Vec3fa v1(vertex_positions[3 * triangle_indices[3 * primID + 1] + 0],
                  vertex_positions[3 * triangle_indices[3 * primID + 1] + 1],
                  vertex_positions[3 * triangle_indices[3 * primID + 1] + 2]);
        Vec3fa v2(vertex_positions[3 * triangle_indices[3 * primID + 2] + 0],
                  vertex_positions[3 * triangle_indices[3 * primID + 2] + 1],
                  vertex_positions[3 * triangle_indices[3 * primID + 2] + 2]);

        // Determine distance to closest point on triangle
        float u, v;
        const Vec3fa p = closestPointTriangle(q, v0, v1, v2, u, v);
        float d = (q - p).norm();

        // Store result in userPtr and update the query radius if we found a
        // point closer to the query position. This is optional but allows for
        // faster traversal (due to better culling).
        if (d < args->query->radius) {
            args->query->radius = d;
            result->p = p;
            result->primID = primID;
            result->geomID = geomID;
            Vec3fa e1 = v1 - v0;
            Vec3fa e2 = v2 - v0;
            result->uv = Vec2f(u, v);
            result->n = (e1.cross(e2)).normalized();
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
    // The maximum number of rays used in calls to embree.
    const size_t BATCH_SIZE = 1024;
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
                  float* primitive_normals,
                  const int nthreads) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        struct RTCIntersectContext context;
        rtcInitIntersectContext(&context);

        auto LoopFn = [&](const tbb::blocked_range<size_t>& range) {
            std::vector<RTCRayHit> rayhits(range.size());

            for (size_t i = range.begin(); i < range.end(); ++i) {
                RTCRayHit& rh = rayhits[i - range.begin()];
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
                rh.ray.id = i - range.begin();
                rh.ray.flags = 0;
                rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            }

            rtcIntersect1M(scene_, &context, &rayhits[0], range.size(),
                           sizeof(RTCRayHit));

            for (size_t i = range.begin(); i < range.end(); ++i) {
                RTCRayHit rh = rayhits[i - range.begin()];
                size_t idx = rh.ray.id + range.begin();
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
        };

        if (nthreads > 0) {
            tbb::task_arena arena(nthreads);
            arena.execute([&]() {
                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                        LoopFn);
            });
        } else {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                    LoopFn);
        }
    }

    void TestOcclusions(const float* const rays,
                        const size_t num_rays,
                        const float tnear,
                        const float tfar,
                        int8_t* occluded,
                        const int nthreads) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        struct RTCIntersectContext context;
        rtcInitIntersectContext(&context);

        auto LoopFn = [&](const tbb::blocked_range<size_t>& range) {
            std::vector<RTCRay> rayvec(range.size());
            for (size_t i = range.begin(); i < range.end(); ++i) {
                RTCRay& ray = rayvec[i - range.begin()];
                const float* r = &rays[i * 6];
                ray.org_x = r[0];
                ray.org_y = r[1];
                ray.org_z = r[2];
                ray.dir_x = r[3];
                ray.dir_y = r[4];
                ray.dir_z = r[5];
                ray.tnear = tnear;
                ray.tfar = tfar;
                ray.mask = 0;
                ray.id = i - range.begin();
                ray.flags = 0;
            }

            rtcOccluded1M(scene_, &context, &rayvec[0], range.size(),
                          sizeof(RTCRay));

            for (size_t i = range.begin(); i < range.end(); ++i) {
                RTCRay ray = rayvec[i - range.begin()];
                size_t idx = ray.id + range.begin();
                occluded[idx] = int8_t(
                        -std::numeric_limits<float>::infinity() == ray.tfar);
            }
        };

        if (nthreads > 0) {
            tbb::task_arena arena(nthreads);
            arena.execute([&]() {
                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                        LoopFn);
            });
        } else {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                    LoopFn);
        }
    }

    void CountIntersections(const float* const rays,
                            const size_t num_rays,
                            int* intersections,
                            const int nthreads) {
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

        auto LoopFn = [&](const tbb::blocked_range<size_t>& range) {
            std::vector<RTCRayHit> rayhits(range.size());

            for (size_t i = range.begin(); i < range.end(); ++i) {
                RTCRayHit* rh = &rayhits[i - range.begin()];
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
            rtcIntersect1M(scene_, &context.context, &rayhits[0], range.size(),
                           sizeof(RTCRayHit));
        };

        if (nthreads > 0) {
            tbb::task_arena arena(nthreads);
            arena.execute([&]() {
                tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                        LoopFn);
            });
        } else {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, num_rays, BATCH_SIZE),
                    LoopFn);
        }
    }

    void ComputeClosestPoints(const float* const query_points,
                              const size_t num_query_points,
                              float* closest_points,
                              unsigned int* geometry_ids,
                              unsigned int* primitive_ids,
                              float* primitive_uvs,
                              float* primitive_normals,
                              const int nthreads) {
        if (!scene_committed_) {
            rtcCommitScene(scene_);
            scene_committed_ = true;
        }

        auto LoopFn = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
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

                closest_points[3 * i + 0] = result.p.x();
                closest_points[3 * i + 1] = result.p.y();
                closest_points[3 * i + 2] = result.p.z();
                geometry_ids[i] = result.geomID;
                primitive_ids[i] = result.primID;
                primitive_uvs[2 * i + 0] = result.uv.x();
                primitive_uvs[2 * i + 1] = result.uv.y();
                primitive_normals[3 * i + 0] = result.n.x();
                primitive_normals[3 * i + 1] = result.n.y();
                primitive_normals[3 * i + 2] = result.n.z();
            }
        };

        if (nthreads > 0) {
            tbb::task_arena arena(nthreads);
            arena.execute([&]() {
                tbb::parallel_for(tbb::blocked_range<size_t>(
                                          0, num_query_points, BATCH_SIZE),
                                  LoopFn);
            });
        } else {
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, num_query_points, BATCH_SIZE),
                    LoopFn);
        }
    }
};

RaycastingScene::RaycastingScene(int64_t nthreads)
    : impl_(new RaycastingScene::Impl()) {
    if (nthreads > 0) {
        std::string config("threads=" + std::to_string(nthreads));
        impl_->device_ = rtcNewDevice(config.c_str());
    } else {
        impl_->device_ = rtcNewDevice(NULL);
    }
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

uint32_t RaycastingScene::AddTriangles(const core::Tensor& vertex_positions,
                                       const core::Tensor& triangle_indices) {
    core::AssertTensorDevice(vertex_positions, impl_->tensor_device_);
    core::AssertTensorShape(vertex_positions, {utility::nullopt, 3});
    core::AssertTensorDtype(vertex_positions, core::Float32);
    core::AssertTensorDevice(triangle_indices, impl_->tensor_device_);
    core::AssertTensorShape(triangle_indices, {utility::nullopt, 3});
    core::AssertTensorDtype(triangle_indices, core::UInt32);

    const size_t num_vertices = vertex_positions.GetLength();
    const size_t num_triangles = triangle_indices.GetLength();

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
        auto data = vertex_positions.Contiguous();
        memcpy(vertex_buffer, data.GetDataPtr(),
               sizeof(float) * 3 * num_vertices);
    }
    {
        auto data = triangle_indices.Contiguous();
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
    size_t num_verts = mesh.GetVertexPositions().GetLength();
    if (num_verts > std::numeric_limits<uint32_t>::max()) {
        utility::LogError(
                "Cannot add mesh with more than {} vertices to the scene",
                std::numeric_limits<uint32_t>::max());
    }
    return AddTriangles(mesh.GetVertexPositions(),
                        mesh.GetTriangleIndices().To(core::UInt32));
}

std::unordered_map<std::string, core::Tensor> RaycastingScene::CastRays(
        const core::Tensor& rays, const int nthreads) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(rays, "rays", 6,
                                                 impl_->tensor_device_);
    auto shape = rays.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_rays = shape.NumElements();

    std::unordered_map<std::string, core::Tensor> result;
    result["t_hit"] = core::Tensor(shape, core::Float32);
    result["geometry_ids"] = core::Tensor(shape, core::UInt32);
    result["primitive_ids"] = core::Tensor(shape, core::UInt32);
    shape.push_back(2);
    result["primitive_uvs"] = core::Tensor(shape, core::Float32);
    shape.back() = 3;
    result["primitive_normals"] = core::Tensor(shape, core::Float32);

    auto data = rays.Contiguous();
    impl_->CastRays<false>(data.GetDataPtr<float>(), num_rays,
                           result["t_hit"].GetDataPtr<float>(),
                           result["geometry_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_ids"].GetDataPtr<uint32_t>(),
                           result["primitive_uvs"].GetDataPtr<float>(),
                           result["primitive_normals"].GetDataPtr<float>(),
                           nthreads);

    return result;
}

core::Tensor RaycastingScene::TestOcclusions(const core::Tensor& rays,
                                             const float tnear,
                                             const float tfar,
                                             const int nthreads) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(rays, "rays", 6,
                                                 impl_->tensor_device_);
    auto shape = rays.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_rays = shape.NumElements();

    core::Tensor result(shape, core::Bool);

    auto data = rays.Contiguous();
    impl_->TestOcclusions(data.GetDataPtr<float>(), num_rays, tnear, tfar,
                          reinterpret_cast<int8_t*>(result.GetDataPtr<bool>()),
                          nthreads);

    return result;
}

core::Tensor RaycastingScene::CountIntersections(const core::Tensor& rays,
                                                 const int nthreads) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(rays, "rays", 6,
                                                 impl_->tensor_device_);
    auto shape = rays.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_rays = shape.NumElements();

    core::Tensor intersections(shape, core::Dtype::FromType<int>());

    auto data = rays.Contiguous();

    impl_->CountIntersections(data.GetDataPtr<float>(), num_rays,
                              intersections.GetDataPtr<int>(), nthreads);
    return intersections;
}

std::unordered_map<std::string, core::Tensor>
RaycastingScene::ComputeClosestPoints(const core::Tensor& query_points,
                                      const int nthreads) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();

    std::unordered_map<std::string, core::Tensor> result;
    result["geometry_ids"] = core::Tensor(shape, core::UInt32);
    result["primitive_ids"] = core::Tensor(shape, core::UInt32);
    shape.push_back(3);
    result["points"] = core::Tensor(shape, core::Float32);
    result["primitive_normals"] = core::Tensor(shape, core::Float32);
    shape.back() = 2;
    result["primitive_uvs"] = core::Tensor(shape, core::Float32);

    auto data = query_points.Contiguous();
    impl_->ComputeClosestPoints(data.GetDataPtr<float>(), num_query_points,
                                result["points"].GetDataPtr<float>(),
                                result["geometry_ids"].GetDataPtr<uint32_t>(),
                                result["primitive_ids"].GetDataPtr<uint32_t>(),
                                result["primitive_uvs"].GetDataPtr<float>(),
                                result["primitive_normals"].GetDataPtr<float>(),
                                nthreads);

    return result;
}

core::Tensor RaycastingScene::ComputeDistance(const core::Tensor& query_points,
                                              const int nthreads) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.

    auto data = query_points.Contiguous();
    auto closest_points = ComputeClosestPoints(data, nthreads);

    size_t num_query_points = shape.NumElements();
    Eigen::Map<Eigen::MatrixXf> query_points_map(data.GetDataPtr<float>(), 3,
                                                 num_query_points);
    Eigen::Map<Eigen::MatrixXf> closest_points_map(
            closest_points["points"].GetDataPtr<float>(), 3, num_query_points);
    core::Tensor distance(shape, core::Float32);
    Eigen::Map<Eigen::VectorXf> distance_map(distance.GetDataPtr<float>(),
                                             num_query_points);

    distance_map = (closest_points_map - query_points_map).colwise().norm();
    return distance;
}

namespace {
// Helper function to determine the inside and outside with voting.
core::Tensor VoteInsideOutside(RaycastingScene& scene,
                               const core::Tensor& query_points,
                               const int nthreads = 0,
                               const int num_votes = 3,
                               const int inside_val = 1,
                               const int outside_val = 0) {
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();

    // Use local RNG here to generate rays with a similar direction in a
    // deterministic manner.
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.001, 0.001);
    Eigen::MatrixXf ray_dirs(3, num_votes);
    ray_dirs = ray_dirs.unaryExpr([&](float) { return 1 + dist(gen); });

    auto query_points_ = query_points.Contiguous();
    Eigen::Map<Eigen::MatrixXf> query_points_map(
            query_points_.GetDataPtr<float>(), 3, num_query_points);

    core::Tensor rays({int64_t(num_votes * num_query_points), 6},
                      core::Float32);
    Eigen::Map<Eigen::MatrixXf> rays_map(rays.GetDataPtr<float>(), 6,
                                         num_votes * num_query_points);
    if (num_votes > 1) {
        for (size_t i = 0; i < num_query_points; ++i) {
            for (int j = 0; j < num_votes; ++j) {
                rays_map.col(i * num_votes + j).topRows<3>() =
                        query_points_map.col(i);
                rays_map.col(i * num_votes + j).bottomRows<3>() =
                        ray_dirs.col(j);
            }
        }
    } else {
        for (size_t i = 0; i < num_query_points; ++i) {
            rays_map.col(i).topRows<3>() = query_points_map.col(i);
            rays_map.col(i).bottomRows<3>() = ray_dirs;
        }
    }

    auto intersections = scene.CountIntersections(rays, nthreads);
    Eigen::Map<Eigen::MatrixXi> intersections_map(
            intersections.GetDataPtr<int>(), num_votes, num_query_points);

    if (num_votes > 1) {
        core::Tensor result({int64_t(num_query_points)}, core::Int32);
        Eigen::Map<Eigen::VectorXi> result_map(result.GetDataPtr<int>(),
                                               num_query_points);
        result_map =
                intersections_map.unaryExpr([&](const int x) { return x % 2; })
                        .colwise()
                        .sum()
                        .unaryExpr([&](const int x) {
                            return (x > num_votes / 2) ? inside_val
                                                       : outside_val;
                        });
        return result.Reshape(shape);
    } else {
        intersections_map = intersections_map.unaryExpr([&](const int x) {
            return (x % 2) ? inside_val : outside_val;
        });
        return intersections.Reshape(shape);
    }
}
}  // namespace

core::Tensor RaycastingScene::ComputeSignedDistance(
        const core::Tensor& query_points,
        const int nthreads,
        const int nsamples) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);

    if (nsamples < 1 || (nsamples % 2) != 1) {
        open3d::utility::LogError("nsamples must be odd and >= 1 but is {}",
                                  nsamples);
    }
    auto shape = query_points.GetShape();
    shape.pop_back();  // Remove last dim, we want to use this shape for the
                       // results.
    size_t num_query_points = shape.NumElements();
    auto data = query_points.Contiguous();
    auto distance = ComputeDistance(data, nthreads);
    Eigen::Map<Eigen::VectorXf> distance_map(distance.GetDataPtr<float>(),
                                             num_query_points);

    auto inside_outside =
            VoteInsideOutside(*this, data, nthreads, nsamples, -1, 1);
    Eigen::Map<Eigen::VectorXi> inside_outside_map(
            inside_outside.GetDataPtr<int>(), num_query_points);
    distance_map.array() *= inside_outside_map.array().cast<float>();
    return distance;
}

core::Tensor RaycastingScene::ComputeOccupancy(const core::Tensor& query_points,
                                               const int nthreads,
                                               const int nsamples) {
    AssertTensorDtypeLastDimDeviceMinNDim<float>(query_points, "query_points",
                                                 3, impl_->tensor_device_);

    if (nsamples < 1 || (nsamples % 2) != 1) {
        open3d::utility::LogError("samples must be odd and >= 1 but is {}",
                                  nsamples);
    }

    auto result = VoteInsideOutside(*this, query_points, nthreads, nsamples);
    return result.To(core::Float32);
}

core::Tensor RaycastingScene::CreateRaysPinhole(
        const core::Tensor& intrinsic_matrix,
        const core::Tensor& extrinsic_matrix,
        int width_px,
        int height_px) {
    core::AssertTensorDevice(intrinsic_matrix, core::Device());
    core::AssertTensorShape(intrinsic_matrix, {3, 3});
    core::AssertTensorDevice(extrinsic_matrix, core::Device());
    core::AssertTensorShape(extrinsic_matrix, {4, 4});

    auto intrinsic_matrix_contig =
            intrinsic_matrix.To(core::Float64).Contiguous();
    auto extrinsic_matrix_contig =
            extrinsic_matrix.To(core::Float64).Contiguous();
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

    core::Tensor rays({height_px, width_px, 6}, core::Float32);
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
    core::AssertTensorDevice(center, core::Device());
    core::AssertTensorShape(center, {3});
    core::AssertTensorDevice(eye, core::Device());
    core::AssertTensorShape(eye, {3});
    core::AssertTensorDevice(up, core::Device());
    core::AssertTensorShape(up, {3});

    double focal_length =
            0.5 * width_px / std::tan(0.5 * (M_PI / 180) * fov_deg);

    core::Tensor intrinsic_matrix =
            core::Tensor::Eye(3, core::Float64, core::Device());
    Eigen::Map<Eigen::MatrixXd> intrinsic_matrix_map(
            intrinsic_matrix.GetDataPtr<double>(), 3, 3);
    intrinsic_matrix_map(0, 0) = focal_length;
    intrinsic_matrix_map(1, 1) = focal_length;
    intrinsic_matrix_map(2, 0) = 0.5 * width_px;
    intrinsic_matrix_map(2, 1) = 0.5 * height_px;

    auto center_contig = center.To(core::Float64).Contiguous();
    auto eye_contig = eye.To(core::Float64).Contiguous();
    auto up_contig = up.To(core::Float64).Contiguous();

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
            core::Tensor::Eye(4, core::Float64, core::Device());
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
