// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared Gaussian splatting geometry + composite dispatch sequence. OpenGL and
// Metal backends supply a GaussianSplatGpuContext implementation.

#pragma once

#include <cstdint>
#include <vector>

#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

struct PackedGaussianScene;

/// Resize/upload buffers, then run the projection → radix → payload chain.
/// @param scene_change_id   Scene geometry change token from Filament.
/// @param scene_changed     True when splat buffers must be re-uploaded.
bool RunGaussianGeometryPasses(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        const PackedGaussianScene& packed,
        const std::vector<GaussianSplatRenderer::PassDispatch>& dispatches,
        GaussianSplatViewGpuResources& vs,
        std::uint64_t scene_change_id,
        std::uint32_t source_splat_count,
        bool scene_changed);

/// Final composite pass into imported color/depth targets.
bool RunGaussianCompositePass(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        const std::vector<GaussianSplatRenderer::PassDispatch>& dispatches,
        GaussianSplatViewGpuResources& vs,
        GaussianSplatRenderer::OutputTargets& targets);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
