// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Stub implementation of RaycastingScene for platforms where Embree is
// unavailable (e.g. Windows ARM64).  All methods log an error at runtime.
// ----------------------------------------------------------------------------

#include <limits>
#include <unordered_map>

#include "open3d/t/geometry/RaycastingScene.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {

// Provide a complete definition of Impl so that std::unique_ptr<Impl> can be
// destroyed without requiring the full Embree-based implementation.
struct RaycastingScene::Impl {
    // Empty stub – no Embree state.
};

// ctor: do NOT call LogError here so that objects can be constructed without
// immediately aborting (TriangleMesh creates one internally).
RaycastingScene::RaycastingScene(int64_t /*nthreads*/,
                                 const core::Device& /*device*/)
    : impl_(std::make_unique<Impl>()) {}

// dtor: Impl is now complete, so default destruction works.
RaycastingScene::~RaycastingScene() = default;

uint32_t RaycastingScene::AddTriangles(
        const core::Tensor& /*vertex_positions*/,
        const core::Tensor& /*triangle_indices*/) {
    utility::LogError(
            "RaycastingScene::AddTriangles is unavailable: Embree is disabled "
            "(OPEN3D_DISABLE_EMBREE=1).");
}

uint32_t RaycastingScene::AddTriangles(const TriangleMesh& /*mesh*/) {
    utility::LogError(
            "RaycastingScene::AddTriangles is unavailable: Embree is disabled "
            "(OPEN3D_DISABLE_EMBREE=1).");
}

std::unordered_map<std::string, core::Tensor> RaycastingScene::CastRays(
        const core::Tensor& /*rays*/, const int /*nthreads*/) const {
    utility::LogError(
            "RaycastingScene::CastRays is unavailable: Embree is disabled "
            "(OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::TestOcclusions(const core::Tensor& /*rays*/,
                                             const float /*tnear*/,
                                             const float /*tfar*/,
                                             const int /*nthreads*/) {
    utility::LogError(
            "RaycastingScene::TestOcclusions is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::CountIntersections(const core::Tensor& /*rays*/,
                                                 const int /*nthreads*/) {
    utility::LogError(
            "RaycastingScene::CountIntersections is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

std::unordered_map<std::string, core::Tensor>
RaycastingScene::ListIntersections(const core::Tensor& /*rays*/,
                                   const int /*nthreads*/) {
    utility::LogError(
            "RaycastingScene::ListIntersections is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

std::unordered_map<std::string, core::Tensor>
RaycastingScene::ComputeClosestPoints(const core::Tensor& /*query_points*/,
                                      const int /*nthreads*/) {
    utility::LogError(
            "RaycastingScene::ComputeClosestPoints is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::ComputeDistance(
        const core::Tensor& /*query_points*/, const int /*nthreads*/) {
    utility::LogError(
            "RaycastingScene::ComputeDistance is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::ComputeSignedDistance(
        const core::Tensor& /*query_points*/,
        const int /*nthreads*/,
        const int /*nsamples*/) {
    utility::LogError(
            "RaycastingScene::ComputeSignedDistance is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::ComputeOccupancy(
        const core::Tensor& /*query_points*/,
        const int /*nthreads*/,
        const int /*nsamples*/) {
    utility::LogError(
            "RaycastingScene::ComputeOccupancy is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::CreateRaysPinhole(
        const core::Tensor& /*intrinsic_matrix*/,
        const core::Tensor& /*extrinsic_matrix*/,
        int /*width_px*/,
        int /*height_px*/) {
    utility::LogError(
            "RaycastingScene::CreateRaysPinhole is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

core::Tensor RaycastingScene::CreateRaysPinhole(double /*fov_deg*/,
                                                const core::Tensor& /*center*/,
                                                const core::Tensor& /*eye*/,
                                                const core::Tensor& /*up*/,
                                                int /*width_px*/,
                                                int /*height_px*/) {
    utility::LogError(
            "RaycastingScene::CreateRaysPinhole is unavailable: Embree is "
            "disabled (OPEN3D_DISABLE_EMBREE=1).");
}

uint32_t RaycastingScene::INVALID_ID() {
    return std::numeric_limits<uint32_t>::max();
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d