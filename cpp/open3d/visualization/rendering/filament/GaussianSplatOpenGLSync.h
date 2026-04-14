// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Thin OpenGL fence helpers for cross-context synchronization.
//
// These helpers are intentionally isolated from the rest of the GS pipeline
// so that GL-specific headers (GL/glew.h) do not leak into translation units
// that do not dispatch GL work.
//
// Usage pattern (shared-depth handoff):
//   // Filament context (inserting context):
//   std::uintptr_t fence = CreateAndFlushOpenGLFence();
//   ...store fence in OutputTargets.scene_depth_ready_fence...
//
//   // Compute shared context (waiting context), before composite dispatch:
//   WaitOpenGLFenceOnGpu(fence);
//   DestroyOpenGLFence(fence);

#pragma once

#if !defined(__APPLE__)

#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

/// Create a GLsync fence on the current GL context and immediately flush so
/// the fence command reaches the GPU before a waiting context inspects it.
/// Both steps are bundled here: omitting the flush causes a hang when the
/// waiting context uses glWaitSync without a matching glFlush on the inserter.
/// Returns an opaque handle (cast from GLsync), or 0 on failure.
[[nodiscard]] std::uintptr_t CreateAndFlushOpenGLFence();

/// GPU-side wait on the fence (no CPU stall).  Calls glWaitSync with
/// GL_TIMEOUT_IGNORED so the GPU waits without blocking the main thread.
/// fence must be a handle returned by CreateAndFlushOpenGLFence().
void WaitOpenGLFenceOnGpu(std::uintptr_t fence);

/// Destroy a fence previously created by CreateAndFlushOpenGLFence().
/// Must be called from a context that shares the same GL object namespace
/// as the one that created the fence.
void DestroyOpenGLFence(std::uintptr_t fence);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
