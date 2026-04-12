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
struct GaussianSplatPackedAttrs;

/// Resize/upload buffers, then run the projection → radix → payload chain.
/// Dispatch grid sizes are computed inline from frame_data and config.
/// @param frame          Per-frame view parameters (UBO, splat/tile counts).
/// @param attrs          Pre-packed per-splat GPU data (uploaded when
/// scene_changed).
/// @param scene_change_id  Scene geometry change token from Filament.
/// @param scene_changed  True when per-splat buffers must be re-uploaded.
bool RunGaussianGeometryPasses(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        const PackedGaussianScene& frame_data,
        const GaussianSplatPackedAttrs& attrs,
        GaussianSplatViewGpuResources& vs,
        std::uint64_t scene_change_id,
        bool scene_changed);

/// Final composite pass into imported color/depth targets.
/// Dispatch grid sizes are computed inline from config and targets dimensions.
bool RunGaussianCompositePass(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        GaussianSplatViewGpuResources& vs,
        GaussianSplatRenderer::OutputTargets& targets);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
