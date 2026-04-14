// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if !defined(__APPLE__)

#include "open3d/visualization/rendering/filament/GaussianSplatOpenGLSync.h"

#include <GL/glew.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

std::uintptr_t CreateAndFlushOpenGLFence() {
    // glFenceSync inserts a sync object into the server command stream.
    // The subsequent glFlush is REQUIRED before a waiting context can use
    // glWaitSync: without it the sync command may still be buffered client-side
    // and the waiting context will hang indefinitely.
    GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!fence) {
        utility::LogWarning(
                "GaussianSplat: glFenceSync failed (GL error={:#x})",
                glGetError());
        return 0;
    }
    glFlush();  // Push the fence command to the GPU before the waiter sees it.
    return reinterpret_cast<std::uintptr_t>(fence);
}

void WaitOpenGLFenceOnGpu(std::uintptr_t fence) {
    if (fence == 0) return;
    // glWaitSync stalls the GPU (not the CPU) until the fence is signalled.
    // GL_TIMEOUT_IGNORED: no timeout — the GPU waits as long as needed.
    glWaitSync(reinterpret_cast<GLsync>(fence), 0, GL_TIMEOUT_IGNORED);
}

void DestroyOpenGLFence(std::uintptr_t fence) {
    if (fence == 0) return;
    glDeleteSync(reinterpret_cast<GLsync>(fence));
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
