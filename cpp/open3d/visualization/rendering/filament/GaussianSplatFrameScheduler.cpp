// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianSplatFrameScheduler.h"

#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {
namespace GaussianSplatFrameScheduler {

bool CompositeRunsAfterFilamentEndFrame() {
#if defined(__APPLE__)
    // Metal: endFrame() commits Filament's command buffer so the depth texture
    // is guaranteed ready before our composite command buffer executes on the
    // same queue.
    return true;
#else
    return false;
#endif
}

void RunGeometry(GaussianSplatRenderer& gs_renderer,
                 FilamentView& view,
                 FilamentScene& scene) {
    gs_renderer.RenderGeometryStage(view, scene);
}

void RunComposite(GaussianSplatRenderer& gs_renderer, FilamentView& view) {
    (void)gs_renderer.RenderCompositeStage(view);
}

}  // namespace GaussianSplatFrameScheduler
}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
