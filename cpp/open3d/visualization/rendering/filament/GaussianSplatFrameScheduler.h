// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Unified per-frame scheduling helpers for Gaussian splatting.
//
// Both the interactive (FilamentRenderer) and offscreen (FilamentRenderToBuffer)
// paths use the same Apple vs non-Apple ordering policy for geometry and
// composite stages.  Centralising the logic here prevents the two callers from
// drifting out of sync.

#pragma once

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentScene;
class FilamentView;
class GaussianSplatRenderer;

namespace GaussianSplatFrameScheduler {

/// Returns true when the composite pass must run AFTER Filament's endFrame().
/// On Apple (Metal) the depth texture is not guaranteed to be fully written
/// until the Metal command buffer commits at endFrame().  On non-Apple (OpenGL)
/// the composite can run during the frame before endFrame().
bool CompositeRunsAfterFilamentEndFrame();

/// Run the Gaussian splat geometry pass for a single (view, scene) pair.
/// The caller is responsible for ensuring Filament's prior work has been
/// flushed (pre-geometry flushAndWait on non-Apple) before the first call
/// in a batch.
void RunGeometry(GaussianSplatRenderer& gs_renderer,
                 FilamentView& view,
                 FilamentScene& scene);

/// Run the Gaussian splat composite pass for a single view.
/// Must be called at the platform-appropriate point relative to endFrame():
///   non-Apple — before endFrame() (inside the render loop)
///   Apple     — after  endFrame() (outside the render loop)
void RunComposite(GaussianSplatRenderer& gs_renderer, FilamentView& view);

}  // namespace GaussianSplatFrameScheduler
}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
