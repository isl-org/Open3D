# 3D Gaussian Splatting Rendering in Open3D — Design & Implementation

## Overview

Open3D supports real-time 3D Gaussian Splatting (3DGS) rendering through a GPU compute pipeline
that runs alongside the Filament-based visualization engine. The compute pipeline projects,
sorts, and composites Gaussian splats into a color image. A shared GL depth texture allows
the composite shader to reject splats behind Filament-rendered mesh geometry, producing
correct per-splat depth occlusion.

**Supported platforms**: Linux X11/GLX (operational, including Wayland via XWayland),
Windows/WGL (planned), macOS/Metal (placeholder only).

---

## Architecture

### Rendering Pipeline

The pipeline is split into two stages: a heavy geometry stage (projection + sort) that can
overlap with Filament mesh rasterization on the GPU, and a lightweight composite stage that
runs after Filament completes (because it needs the scene depth).

```
┌─────────────────────────────────────────────────────────────────┐
│  BeginFrame (FilamentRenderer::BeginFrame)                      │
│                                                                 │
│  1. engine_.flushAndWait()   — Filament GPU idle                │
│  2. GS Stage A (geometry)    — 4-pass GPU compute               │
│     ├─ Projection            — splat → screen space             │
│     ├─ Tile Prefix Sum       — per-tile entry counts            │
│     ├─ Tile Scatter          — splats → tile buckets            │
│     └─ Radix Sort (4 passes) — sort by (tile_id<<16|depth>>16)  │
│     (no glFinish — GPU work overlaps with Filament)             │
│  3. renderer_->beginFrame()  — Filament starts frame            │
├─────────────────────────────────────────────────────────────────┤
│  Draw (FilamentRenderer::Draw)                                  │
│                                                                 │
│  4. Filament scene draw      — meshes render into cached view   │
│     (depth writes to shared GL DEPTH_COMPONENT32F texture)      │
│  5. engine_.flushAndWait()   — GPU idle, depth is ready         │
│  6. GS Stage B (composite)   — 1-pass GPU compute               │
│     └─ Composite             — per-splat depth test vs scene    │
│        (writes color to imported GL RGBA16F texture)            │
│  7. GUI draw (ImGui)         — base image + GS overlay          │
└─────────────────────────────────────────────────────────────────┘
```

### Depth-Aware Compositing (Zero-Copy)

The GS compute context and Filament share the same GLX context group. GL textures created
in our context are visible in Filament's context by handle — no copies required.

**Shared depth texture**: `PrepareOutputTargets` creates a `GL_DEPTH_COMPONENT32F` texture,
imports it into Filament as `DEPTH_ATTACHMENT | SAMPLEABLE`, and sets it as the depth
attachment on the view's render target. Filament writes depth during mesh rendering.
After `flushAndWait()`, the composite shader reads this same texture as `sampler2D` at
binding 14 and performs per-splat depth rejection.

**Shared color texture**: A `GL_RGBA16F` texture is imported into Filament as `SAMPLEABLE`.
The composite shader writes to it via `imageStore`. ImGui displays it as an overlay using
standard alpha blending (SrcAlpha, 1-SrcAlpha) over the Filament scene color buffer.

**MSAA constraint**: Filament asserts `!msaa.enabled || !renderTarget->hasSampleableDepth()`
(`View.h:210`). MSAA is disabled for views using the shared sampleable depth texture.
This is acceptable because 3DGS uses Gaussian kernel anti-aliasing.

### Shared GL Context Strategy

The compute context is created **before** Filament's `Engine::create()`. This is the only
window when context sharing can be established — GLX/EGL sharing is set at context creation
time and cannot be added retroactively.

1. `FilamentEngine.cpp` calls `GaussianComputeOpenGLContext::GetInstance().InitializeStandalone()`
2. `InitializeStandalone()` → `InitializeGLXStandalone()` on X11 or `InitializeEGL()` on Wayland
3. The resulting native context handle is passed to `Engine::create()` as `sharedGLContext`
4. Filament's `PlatformGLX::createDriver` sees the shared context, finds a matching FBConfig
   via `GLX_FBCONFIG_ID`, and creates its own context with `glXCreateContextAttribs(…, share)`
5. Both contexts now share the same GL object namespace — `import()` works by handle

The older `InitializeGLX()` path (which tried to query Filament's context *after* creation)
is retained as a fallback but is never reached in normal operation because Filament runs GL
on a dedicated driver thread (so `glXGetCurrentContext()` returns NULL from the main thread).

### Backend Abstraction

```
GaussianComputeRenderer::Backend (abstract)
├── GaussianComputeOpenGLBackend      — Linux + Windows (GL 4.5+ compute, SPIR-V)
├── GaussianComputeMetalBackend       — macOS (placeholder — not functional)
└── GaussianComputePlaceholderBackend — fallback (logs once per view, returns false)
```

Each backend implements `RenderGeometryStage` (passes 1-4) and `RenderCompositeStage`
(pass 5). The split enables GPU overlap: Stage A dispatches compute work without `glFinish()`,
allowing Filament rasterization to execute concurrently. Stage B waits for Filament to
complete (`flushAndWait`) before reading scene depth.

---

## Sort Key Layout

Each TileEntry sort key packs `tile_index` and `depth` into a single 32-bit uint:

```
bits 31..16  tile_index (16 bits, max 65535 tiles)
bits 15..0   depth_key >> 16 (16 bits of depth precision)
```

`key = (tile_index << 16u) | (depth_key >> 16u)`

This supports up to 65,535 tiles — e.g. 4K at 16×16 tile size needs 32,400 tiles.
The previous 12/20 split (`tile << 20`) capped at 4095 tiles and corrupted rendering above
~1280×720. 16 depth bits gives ~0.0015% relative depth precision (sufficient for back-to-front
ordering between neighbouring splats).

The 4-pass radix sort operates on all 32 bits (8 bits per pass), so the key width change
requires no dispatch logic updates.

---

## Completed Work

### PHASE 1: Linux/Windows Default → OpenGL (DONE)
- `FilamentEngine.cpp`: OpenGL is the default for both Linux and Windows.
- macOS uses the default Filament backend (Metal).

### PHASE 2: OpenGL Compute Backend (DONE)
- **GL context** (`GaussianComputeOpenGLContext`): Standalone context created before Filament,
  passed as `sharedGLContext` to `Engine::create()`. Shares GL namespace with Filament's context.
  X11/GLX on X11, EGL on Wayland (runtime-detected via `XDG_SESSION_TYPE`).
- **Pipeline API** (`GaussianComputeOpenGLPipeline`): Thin wrappers around GL 4.5 compute:
  SSBO/UBO bind, image/sampler bind, buffer management, texture management, dispatch, barriers.
  All GL constants as `constexpr` to avoid GL header dependency in consumers.
- **SPIR-V loading**: 9 shaders compiled offline from GLSL `.comp` to Vulkan-targeted SPIR-V
  (`.spv`); loaded via `glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V)` + `glSpecializeShader()`.
  OpenGL SPIR-V target (`-G`) fails for subgroup ops; Vulkan target works via `GL_ARB_gl_spirv`.
- **Radix sort**: 4-pass 8-bit LSD radix sort. Runtime-queries `gl_SubgroupSize`. Sort key:
  `(tile_index << 16) | (depth_key >> 16)` — 16 bits each, max 65535 tiles.
- **Two-stage execution**: `RenderGeometryStage` (passes 1-4) defers `glFinish()` to enable
  overlap. `RenderCompositeStage` (pass 5) runs after `flushAndWait()`.
- **Zero-copy output**: `PrepareOutputTargets` creates GL textures (`DEPTH_COMPONENT32F` +
  `RGBA16F`), imports into Filament via `Texture::Builder::import()`. No CPU staging.
- **Depth-aware compositing**: Composite shader reads scene depth at binding 14, linearizes
  Filament's reversed-Z depth, discards splats behind mesh geometry per-pixel.
- **MSAA disabled** for GS views (required for sampleable depth attachment).
- **Counter readback**: After prefix-sum pass, `total_entries` is read back from a GPU counter
  buffer to size tile entries and sort buffers before scatter.

### PHASE 4: Depth-Based Scene Compositing (DONE)
- `gaussian_composite.comp`: binding 14 = `sampler2D scene_depth`. Reversed-Z linearization.
  Per-splat occlusion test: `if (use_scene_depth && s_depth[i] >= scene_linear_depth) continue`.
- UBO `depth_range_and_flags[3]` set to `1.0` in `RenderCompositeStage` when scene depth valid.
- Example (`GaussianSplat.cpp`): red sphere placed at scene bbox center for depth compositing
  testing.

### PHASE 5: Vulkan Code Removal (DONE)
- `GaussianComputeVulkanPipeline.h/.cpp` deleted.
- `GaussianComputeVulkanBackend` class removed from `GaussianComputeRenderer.cpp`.

### Bug Fixes (DONE)
- **Tile index overflow** (`gaussian_radix_sort_keygen.comp`): key was `tile_index << 20`
  (12-bit tile field, max 4095 tiles). Changed to `tile_index << 16` (16 bits, max 65535).
  Fixed corruption on windows larger than ~1280×720.
- **Use-after-free on resize**: `FilamentView::EnableViewCaching()` freed `color_buffer_`
  while the GS render target still referenced it. Fixed by calling
  `scene_->InvalidateGaussianComputeOutput(*this)` before freeing `color_buffer_`, tearing
  down the GS render target first through:
  `FilamentScene → FilamentRenderer → GaussianComputeRenderer::InvalidateOutputForView()`.
- **Non-sharing contexts** (depth all zeros): two independent GL namespaces meant imported
  texture IDs were meaningless in Filament's context. Fixed by creating the compute context
  before Filament via `InitializeStandalone()` and passing it as `sharedGLContext`.
- `ForEachView` vs `ForEachActiveView`: scene persistence on mouse-move.
- View pruning: only removes outputs for truly deleted views (not merely inactive ones).
- `scene_change_id` self-assignment: stored from scene in `RenderGeometryStage`, not in
  `RenderCompositeStage` where the scene is inaccessible.

### Optimizations Implemented

| # | Name | Description |
|---|------|-------------|
| Opt1 | Persistent buffers | GPU buffers reused across frames if size sufficient |
| Opt2 | Batched submission | Project + prefix + scatter in one GL session |
| Opt3 | Half-float upload | RGBA16F staging, no float32 roundtrip |
| Opt4 | Compact projected struct | 48 bytes (was 96) |
| Opt5 | GPU-side fill/copy | `ClearGLBuffer` / copy without CPU staging |
| Opt6 | Scene/view separation | Static scene data (positions, SH) cached across camera moves |
| Opt7 | Cooperative tile load | Shared-memory batch loading for the composite pass |
| Opt8 | Early culling | Behind-camera and negligible-alpha splats rejected in projection |
| Opt9 | GPU Stage A overlap | Geometry stage defers `glFinish()` so Filament rasterization overlaps |
| Opt10 | Zero-copy depth | Shared GL depth texture; no CPU readback or re-upload |

### Data Packing (`GaussianComputeDataPacking`)
- `PackGaussianSceneInputs`: CPU → GPU-compatible std140/std430 layouts
- `PackedGaussianViewParams` (std140 UBO): matrices, viewport, scene params.
  `depth_range_and_flags[3]` = scene depth flag (0.0 in geometry stage, 1.0 in composite).
- `PackedProjectedGaussian`: 48-byte per-splat descriptor
- `PackedTileEntry`: 16-byte per tile-entry sort record (depth_key, splat_index, stable_index,
  tile_index stored in the `reserved` field for radix keygen)

---

## Planned Work

### PHASE 3: macOS / Metal Backend

The Metal backend currently has a placeholder stub that logs a message and returns false.
Required work:

1. Port the 9-shader radix sort dispatch logic to Metal (Metal Shading Language)
2. Implement Metal equivalents of `PrepareOutputTargets`: create `MTLTexture` + import via
   Filament's Metal `import()` path
3. Implement `RenderCompositeStage` with scene depth binding
4. Verify context sharing — check if Filament's Metal backend exposes a shared `MTLDevice`
5. Remove the CPU readback path (`DownloadMetalSharedBuffer` + `UploadOutputTextures`)

**Files to modify:**
- `GaussianComputeRenderer.cpp` — `GaussianComputeMetalBackend` class
- `GaussianComputeMetalShaders.h/.mm` — Metal compute dispatch

### PHASE 6: Build Integration Cleanup

- Remove stale `resources/gaussian_compute/` copies (now generated from `shaders/` by CMake)
- Add GL 4.5 minimum capability check to CMake (`GL_ARB_compute_shader`,
  `GL_ARB_shader_storage_buffer_object`, `GL_KHR_shader_subgroup`)
- Verify SPIR-V shader compilation across all supported glslangValidator versions
- Add Python bindings for `GaussianComputeRenderer::SetRenderConfig()` and `SetEnabled()`

---

## Future Work

### FW1: Performance Improvements

| # | Optimization | Impact | Description |
|---|---|---|---|
| FW1-1 | Eliminate counter readback | High | After prefix-sum, `total_entries` is read back via `glGetBufferSubData` — CPU/GPU sync stall. Pre-allocate tile entries buffer to worst-case size and scatter directly without readback. Remove `DownloadGLBuffer(counters_buf)` from the hot path. |
| FW1-2 | Reduce radix sort passes | Medium | Key is 32 bits but only `ceil(log2(tile_count)) + 16` bits are significant. At 4K (32,400 tiles), only 30 bits matter. Running fewer sort passes can reduce one full sort phase per frame. |
| FW1-3 | Fused prefix-sum + scatter | Medium | Merge tile count accumulation and scatter using subgroup prefix sums plus global atomics. Eliminates one compute dispatch and one full barrier round-trip. |
| FW1-4 | Async compute overlap | Medium | Stage A currently serializes with `flushAndWait()` before compute. Move Stage A onto async compute where possible and synchronize with fences/timeline semaphores. |
| FW1-5 | SH degree LOD | Low–Medium | Dynamically reduce SH degree for distant/small splats to reduce bandwidth and projection cost. |
| FW1-6 | Per-tile entry budget | Low | Cap tiles-per-splat to bound worst-case sort/composite cost for huge splats. |
| FW1-7 | Indirect dispatch for scatter | Low | Use GPU-written dispatch arguments (`glDispatchComputeIndirect`) to reduce overhead on sparse scenes. |
| FW1-8 | MSAA re-enable via depth resolve | Low | Re-enable MSAA for meshes via explicit depth resolve while keeping sampleable depth for GS composite. |

### FW2: Native Wayland / EGL (Requires Filament rebuild)

**Current state**: Wayland sessions work via XWayland. `GLFWWindowSystem::Initialize()` forces
`GLFW_PLATFORM_X11` and `GaussianComputeOpenGLContext::InitializeStandalone()` always creates a
GLX context. Both match Filament v1.54.0's compile-time `PlatformGLX` selection.

**Why native EGL requires a Filament rebuild**: `PlatformFactory.cpp` selects `PlatformGLX` or
`PlatformEGLHeadless` at compile time. `PlatformEGLHeadless` has no windowed swapchain. There is
no runtime Wayland platform in Filament v1.54.0.

**Required work for native Wayland (no XWayland):**

1. **Filament platform**: Build Filament from source with
   `-DFILAMENT_SUPPORTS_EGL_ON_LINUX=ON -DFILAMENT_SUPPORTS_XLIB=OFF -DFILAMENT_SUPPORTS_XCB=OFF`
   and add a windowed EGL swapchain to `PlatformEGL` (upstream contribution or local patch).
2. **EGL standalone context**: Implement `InitializeEGLStandalone()` analogous to
   `InitializeGLXStandalone()` — creates an EGL context before Filament for namespace sharing.
3. **Backend selection**: Tie the GS compute backend and the GLFW platform hint to the
   Filament platform chosen at build time (CMake variable or runtime detection of `PlatformEGL`).
4. **Remove X11 override**: Once Filament has a windowed EGL path, remove the
   `GLFW_PLATFORM_X11` hint from `GLFWWindowSystem::Initialize()`.
5. **Validation**: Test with `XDG_SESSION_TYPE=wayland` on Weston/GNOME-Wayland without
   XWayland; verify render correctness and depth compositing.

**Files to modify**: `GaussianComputeOpenGLContext.cpp`, `GLFWWindowSystem.cpp`,
`FilamentEngine.cpp`, `3rdparty/filament/filament_build.cmake`

### FW3: Windows / WGL Platform Testing and Debugging

On Windows, Filament uses WGL for OpenGL. A standalone shared-context setup equivalent to GLX
is still needed to guarantee zero-copy depth sharing.

**Required work:**

1. **WGL standalone context**: Implement `InitializeWGLStandalone()`:
   - Create a hidden 1×1 window + HDC
   - Select pixel format with `wglChoosePixelFormatARB`
   - Create GL 4.5 core `HGLRC` with `wglCreateContextAttribsARB`
   - Return the native handle for `sharedGLContext`
2. **Filament engine hookup**: Extend the shared-context setup path in `FilamentEngine.cpp`
   to run on `_WIN32` as well.
3. **Build plumbing**: Add Windows-specific includes/guards in
   `GaussianComputeOpenGLContext.cpp` and verify linking against OpenGL/WGL symbols.
4. **Validation matrix**: Test Intel/NVIDIA/AMD on Windows 10/11 for:
   shared depth correctness, resize stability, maximize/restore stability, and large-window behavior.
5. **Pixel format matching**: Ensure the WGL pixel format used by our standalone context
   matches what Filament expects for shared context creation.

**Files affected**: `GaussianComputeOpenGLContext.h/.cpp`, `FilamentEngine.cpp`

### FW4: macOS / Metal (see PHASE 3 above)

macOS uses Metal exclusively. See PHASE 3 for the complete implementation plan.

---

## Known Issues / Risks

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| 1 | Counter readback CPU stall | Medium | Remove readback by pre-allocation / indirect dispatch (FW1-1/FW1-7). |
| 2 | Reversed-Z assumption | Medium | Add runtime diagnostics and shader path toggle for non-reversed depth conventions. |
| 3 | Post-processing depth loss | Low–Medium | Keep `SetPostProcessing(false)` for GS views; warn when re-enabled. |
| 4 | Stage A overlap sync | Low | If driver-specific glitches appear, insert `glFlush()` at end of Stage A. |
| 5 | Redundant depth buffer on resize | Low | Avoid creating `depth_buffer_` in cached view when GS shared depth is active. |
| 6 | Native Wayland (no XWayland) unsupported | Low | Filament v1.54.0 has no windowed EGL platform. Current fix forces X11/XWayland. See FW2 for native Wayland plan. |
| 7 | `kProgShellSort` legacy slot | Low | Remove legacy shell-sort shader/program slot after full cross-platform validation. |

---

## File Inventory

### Core implementation files
- `GaussianComputeRenderer.h/.cpp` — Backend interface, OpenGL/Metal backends, output lifecycle;
  `InvalidateOutputForView()` for safe resize
- `GaussianComputeOpenGLPipeline.h/.cpp` — GL 4.5 compute API wrappers
- `GaussianComputeOpenGLContext.h/.cpp` — GLX/EGL context creation; standalone and fallback paths
- `GaussianComputeDataPacking.h/.cpp` — CPU → GPU data packing (std140/std430)
- `GaussianComputeMetalShaders.h/.mm` — Metal compute dispatch (placeholder)
- `FilamentResourceManager.h/.cpp` — `CreateImportedTexture()` for zero-copy import
- `FilamentView.h/.cpp` — `EnableViewCaching()` invalidation fix before freeing color_buffer_
- `FilamentScene.h/.cpp` — `InvalidateGaussianComputeOutput()` forwarding
- `FilamentRenderer.h/.cpp` — frame schedule and GS output forwarding
- `FilamentEngine.cpp` — pre-Filament shared context setup

### Shader files (9 SPIR-V programs)

| Index | File | Pass |
|---|---|---|
| 0 | `gaussian_project.comp` | Pass 1: splat projection, tile rect encoding |
| 1 | `gaussian_prefix_sum.comp` | Pass 2: tile prefix sum, counter write |
| 2 | `gaussian_scatter.comp` | Pass 3: tile-entry scatter |
| 3 | `gaussian_sort.comp` | Legacy shell sort (compiled, not dispatched) |
| 4 | `gaussian_composite.comp` | Pass 5: depth-aware compositing |
| 5 | `gaussian_radix_sort_keygen.comp` | Keygen: `(tile_id<<16)|(depth>>16)` |
| 6 | `gaussian_radix_sort_histograms.comp` | Radix histogram |
| 7 | `gaussian_radix_sort.comp` | Radix scatter |
| 8 | `gaussian_radix_sort_payload.comp` | Payload rearrangement |

### GUI / example files
- `SceneWidget.cpp` — ImGui base image + GS overlay alpha blending
- `examples/cpp/GaussianSplat.cpp` — includes red sphere for depth compositing testing

---

## Verification Checklist

| Test | Command / Steps | Expected Result |
|------|----------------|-----------------|
| Build | `cmake --build build -j$(nproc) --target GaussianSplat` | Zero errors/warnings |
| Basic render | `./bin/examples/GaussianSplat scene.ply` | Scene renders; orbit/zoom stable |
| Depth compositing | Run with red sphere + GS | Correct per-splat occlusion |
| Resize / maximize | Drag/maximize/restore | No panic, no corruption |
| Large window | > 1280×720 (tile_count > 4096) | No tile sort corruption |
| Wayland | `XDG_SESSION_TYPE=wayland ./bin/examples/GaussianSplat ...` | No crash; behavior documented |
| Visibility updates | Toggle geometry show/hide | Immediate updates |
| GL debug | Enable `GL_KHR_debug` callback | No compute dispatch GL errors |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Linux/Windows default → OpenGL | Ensures GS compute works; shared GL context enables zero-copy |
| Standalone context before Filament | GL context sharing is only defined at creation time; Filament context is on driver thread |
| Force GLX on Linux (incl. Wayland) | Filament v1.54.0 uses PlatformGLX unconditionally on Linux. EGL context passed as `sharedGLContext` causes `glXQueryContext` X11 error; Wayland `wl_surface*` passed to `createSwapChain` crashes PlatformGLX. Force `GLFW_PLATFORM_X11` and GLX context; XWayland provides compatibility on Wayland compositors. Native EGL requires a Filament rebuild (see FW2). |
| Two-stage compute split | Enables Stage A overlap with Filament rasterization |
| Compute shader compositing | Per-splat occlusion without extra material pass |
| Shared sampleable depth texture | Zero-copy depth path between Filament and GS composite |
| MSAA disabled for GS views | Required by Filament with sampleable depth attachments |
| Vulkan SPIR-V in OpenGL | Works with subgroup ops where OpenGL SPIR-V target fails |
| Binding 14 reuse | Safe because radix UBO and scene depth sampler are used in disjoint stages |
| Sort key 16/16 split | Supports up to 65535 tiles while preserving usable depth ordering |
| Pre-destroy invalidation on resize | Prevents Filament handle use-after-free during maximize/resize |
