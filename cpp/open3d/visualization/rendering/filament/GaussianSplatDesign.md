# 3D Gaussian Splatting Rendering in Open3D — Design & Implementation

## Overview

Open3D supports real-time 3D Gaussian Splatting (3DGS) rendering through a GPU compute pipeline
that runs alongside the Filament-based visualization engine. The compute pipeline projects,
sorts, and composites Gaussian splats into color and depth images that are then composited with
the normal Filament-rendered scene using depth-based per-pixel selection.

**Supported platforms**: Linux (OpenGL), Windows (OpenGL), macOS (Metal).

---

## Architecture

### Rendering Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Per-frame rendering (FilamentRenderer::BeginFrame)             │
│                                                                 │
│  1. engine_.flushAndWait()   — Filament GPU idle                │
│  2. GS Compute Pipeline      — 5-pass GPU compute              │
│     ├─ Projection            — splat → screen space             │
│     ├─ Tile Prefix Sum       — per-tile entry counts            │
│     ├─ Tile Scatter          — splats → tile buckets            │
│     ├─ Tile Sort             — radix sort by depth              │
│     └─ Composite             — write color + depth images       │
│  3. renderer_->beginFrame()  — Filament renders scene           │
│                                                                 │
│  Compositing (depth-based):                                     │
│  4. Custom Filament material reads both scene depth and         │
│     GS depth, picks nearer pixel for final output               │
└─────────────────────────────────────────────────────────────────┘
```

### Backend Abstraction

The compute pipeline uses an abstract `Backend` interface with platform-specific implementations:

```
GaussianComputeRenderer::Backend (abstract)
├── GaussianComputeOpenGLBackend   — Linux + Windows (GL 4.6 compute)
└── GaussianComputeMetalBackend    — macOS (Metal compute)
```

### Zero-Copy Output

Both backends write compute output to a GPU texture that is directly imported into Filament
via `Texture::Builder::import()`. This eliminates all GPU→CPU→GPU staging transfers (~12 MB/frame
for 1024×768 RGBA16F).

- **OpenGL**: `import((intptr_t)gl_texture_id)` — shared context makes GLuint visible to Filament
- **Metal**: `import((intptr_t)CFBridgingRetain(mtlTexture))` — Metal texture wrapping

---

## Completed Work (committed to git)

### Vulkan/Metal Compute Pipeline (now being migrated to OpenGL)
- 5-pass compute pipeline: projection, prefix sum, scatter, radix sort, composite
- 9 GLSL compute shaders (5 main + 4 radix sort stages)
- GPU-side radix sort with subgroup operations
- Shared-memory cooperative tile loading (Opt7)
- Compact 48-byte ProjectedGaussian struct (Opt4)

### OpenGL Compute Backend (Phase 2 — COMPLETE)
- **GL context**: GLX on X11 (GL 4.6 Core Profile), EGL as Wayland fallback
- **SPIR-V loading**: Vulkan-targeted SPIR-V (`.spv` from `glslangValidator -V --target-env vulkan1.1`)
  loads directly into OpenGL via `glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V)` +
  `glSpecializeShader()`. No separate OpenGL SPIR-V compilation step needed.
- **Radix sort**: 4-pass 8-bit LSD radix sort (keygen → 4×{histogram+scatter} → payload)
  replaces Shell sort that caused GPU timeout on tiles with ~34K entries.
  Uses `gl_SubgroupSize` at runtime (not hardcoded). RadixSortParams at UBO binding 14.
- **All 9 shaders** loaded from SPIR-V; no runtime GLSL compilation.
- **Push constants → UBO**: 4 radix sort shaders converted from Vulkan `push_constant`
  to `layout(std140, binding = 14) uniform RadixSortParams`.
- **Program indices**: `kProgProject(0)..kProgRadixPayload(8)`, `kNumPrograms = 9`.

### Optimizations Implemented
- **Opt1**: Persistent GPU buffers (reuse across frames if size sufficient)
- **Opt2**: Batched command submission (project+prefix+scatter in one session)
- **Opt3**: Half-float direct upload (RGBA16F staging, no float32 roundtrip)
- **Opt4**: Compact projected struct (48 bytes, was 96)
- **Opt5**: GPU-side buffer fill/copy (no CPU staging for clears)
- **Opt6**: Scene/view data separation (scene data persists across camera moves)
- **Opt7**: Shared-memory cooperative tile loading in composite shader
- **Opt8**: Early culling in projection (behind-camera, negligible-alpha reject)

### Data Packing (GaussianComputeDataPacking)
- PackGaussianSceneInputs: CPU → GPU-compatible std140/std430 layouts
- PackedGaussianViewParams: 256-byte uniform block (matrices, viewport, scene params)
- PackedProjectedGaussian: 48-byte per-splat descriptor  
- PackedTileEntry: 16-byte per tile-entry sort record

### Bug Fixes
- ForEachView vs ForEachActiveView: scene persistence on mouse-move
- View pruning: only removes outputs for truly deleted views

---

## Planned Work

### PHASE 1: Switch Linux Default to OpenGL

**Step 1.1**: Change Linux default rendering backend from `kDefault` to `kOpenGL`.

Currently ([FilamentEngine.cpp](cpp/open3d/visualization/rendering/filament/FilamentEngine.cpp)):
```cpp
#ifdef _WIN32
RenderingType EngineInstance::type_ = RenderingType::kOpenGL;  // Windows
#else
RenderingType EngineInstance::type_ = RenderingType::kDefault; // Linux/macOS
#endif
```

Change to:
```cpp
#if defined(_WIN32) || defined(__linux__)
RenderingType EngineInstance::type_ = RenderingType::kOpenGL;  // Windows + Linux
#else
RenderingType EngineInstance::type_ = RenderingType::kDefault; // macOS → Metal
#endif
```

**Wayland/EGL support**: Filament's OpenGL backend on Linux can use either PlatformGLX (X11) or
PlatformEGL (both X11 and Wayland). The Filament prebuilt library includes both platforms.
Filament selects the appropriate platform based on the native window handle:
- X11 window → PlatformGLX (GLX context) or PlatformEGL with EGL_PLATFORM_X11
- Wayland surface → PlatformEGL with EGL_PLATFORM_WAYLAND

Open3D already detects the GLFW platform at runtime ([NativeLinux.cpp](cpp/open3d/visualization/gui/NativeLinux.cpp)):
```cpp
if (glfwGetPlatform() == GLFW_PLATFORM_X11)
    return glfwGetX11Window(window);
else if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
    return glfwGetWaylandWindow(window);
```

GLFW 3.4 (used by Open3D) auto-detects X11 vs Wayland at runtime. No build-time change needed.

**Files**:
- MODIFY: [FilamentEngine.cpp](cpp/open3d/visualization/rendering/filament/FilamentEngine.cpp) —
  lines 35-42, change `#else` → `#elif defined(__APPLE__)`

---

### PHASE 2: OpenGL Compute Backend

**Step 2.1: Shared GL context infrastructure**

Create a GL context that shares resources with Filament's internal context, using EGL
(works on both X11 and Wayland). GLX is X11-only and should be avoided for Wayland
compatibility.

**Approach**: Create an EGL context before Filament engine creation, pass it via
`EngineInstance::SetSharedContext()`. Filament's PlatformEGL creates its context in the
same share group. Both contexts share the GL namespace (textures, SSBOs visible to both).

**Implementation**:
- Detect GLFW platform: if Wayland → EGL with `EGL_PLATFORM_WAYLAND_KHR`; if X11 →
  EGL with `EGL_PLATFORM_X11_KHR` (EGL works on both platforms, avoiding the need for
  separate GLX and EGL code paths)
- On Windows: Use WGL with a dummy HWND, or EGL via ANGLE/Mesa
- Create pbuffer surface (headless) for our compute context
- Call `EngineInstance::SetSharedContext(egl_context)` before `EngineInstance::GetInstance()`

**Threading model**:
- Filament's backend thread owns its GL context
- Our compute context is made current on the main thread
- After `engine_.flushAndWait()`, Filament is idle → safe to dispatch compute
- After compute: `glMemoryBarrier(GL_ALL_BARRIER_BITS)` + `glFlush()`
- Filament's next frame sees writes after implicit sync

**Files**:
- NEW: `GaussianComputeOpenGLContext.h/.cpp` — EGL context creation, make-current, sync
- MODIFY: [FilamentEngine.h/.cpp](cpp/open3d/visualization/rendering/filament/FilamentEngine.h) — integrate shared context setup

**Step 2.2: OpenGL compute pipeline API**

Replace [GaussianComputeVulkanPipeline.h/.cpp](cpp/open3d/visualization/rendering/filament/) with
OpenGL equivalent providing the same free-function C-style API.

| Vulkan Concept | OpenGL Equivalent |
|---|---|
| VkBuffer (SSBO) | GL_SHADER_STORAGE_BUFFER |
| VkBuffer (Uniform) | GL_UNIFORM_BUFFER |
| VkImage (storage) | GL texture + glBindImageTexture() |
| VkPipeline (compute) | GLuint program (compute shader) |
| vkCmdDispatch | glDispatchCompute |
| vkCmdPipelineBarrier | glMemoryBarrier |
| vkCmdFillBuffer | glClearBufferSubData |
| vkCmdCopyBuffer | glCopyBufferSubData |
| push_constants | std140 uniform buffer |
| VkCommandBuffer session | Not needed — GL commands execute immediately |
| vkQueueSubmit + fence | glFenceSync + glClientWaitSync |

**Key simplification**: No command recording/session model. The Vulkan
"BeginSession → Record... → Submit" pattern becomes direct GL calls.

**Files**:
- NEW: `GaussianComputeOpenGLPipeline.h/.cpp`

**Step 2.3: ~~Compile shaders to OpenGL SPIR-V~~ SPIR-V Loading (COMPLETE)**

~~Add a second glslangValidator invocation~~ **Not needed.** Vulkan-targeted SPIR-V
(`-V --target-env vulkan1.1`) loads directly in OpenGL 4.6 via `glShaderBinary(0x9551)` +
`glSpecializeShader("main")`. This was validated at runtime on Mesa Intel LNL.

OpenGL SPIR-V (`-G --target-env opengl`) fails for shaders with subgroup operations
("requires SPIR-V 1.3"), but the Vulkan SPIR-V path works because Mesa's `GL_ARB_gl_spirv`
implementation handles Vulkan SPIR-V binaries correctly.

**Shader source changes** (4 radix sort shaders — DONE):
- Replaced `layout(push_constant, std430) uniform PushConstants { ... };`
  with `layout(std140, binding = 14) uniform RadixSortParams { uint g_num_elements;
  uint g_shift; uint g_num_workgroups; uint g_num_blocks_per_workgroup; };`
- Replaced `#define SUBGROUP_SIZE 32` with runtime `gl_SubgroupSize` in scatter shader
- Expanded `shared uint sums[RADIX_SORT_BINS / SUBGROUP_SIZE]` to `sums[WORKGROUP_SIZE]`
  for variable subgroup sizes

**Runtime loading**: `glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V)` +
`glSpecializeShader()` (GL 4.6) — uses Vulkan SPIR-V directly.

**Files** (DONE):
- NO CHANGES NEEDED: [Open3DAddComputeShaders.cmake](cmake/Open3DAddComputeShaders.cmake) —
  existing Vulkan SPIR-V output works directly
- DONE: 4 radix sort `.comp` shaders — push_constant → uniform block

**Step 2.4: GaussianComputeOpenGLBackend (COMPLETE)**

Implements `Backend` interface with the same 5-pass pipeline architecture:
1. Projection → 2. Prefix Sum → 3. Scatter → 4. Radix Sort → 5. Composite

Radix sort dispatch sequence (10 dispatches total):
```
Keygen: tile_entries → sort_keys[0], sort_values[0]  (composite key = tile_idx<<20 | depth>>12)
for shift in {0, 8, 16, 24}:
    ClearHistogram
    Histogram: sort_keys[src] → histograms           (per-workgroup 256-bin histograms)
    Scatter:   sort_keys[src] → sort_keys[dst]        (8-bit radix scatter with subgroup ops)
               sort_values[src] → sort_values[dst]
    swap src, dst
Payload: sort_values[0] + tile_entries → sorted_entries  (rearrange by sorted indices)
```

Work distribution: `num_wg = ceil(N / (256 * 32))`, `blocks_per_wg = ceil(N / (num_wg * 256))`

Uses OpenGL pipeline API from Step 2.2. ~~Zero-copy output via import() (Step 2.5).~~
Currently uses CPU readback; zero-copy planned.

**Files** (DONE):
- DONE: [GaussianComputeRenderer.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeRenderer.cpp) — OpenGL backend with radix sort, SPIR-V loading

**Step 2.5: Zero-copy output via import()**

1. Create GL texture: `glGenTextures` + `glTexStorage2D(GL_RGBA16F)`
2. Bind as storage image for compute: `glBindImageTexture()`
3. Composite shader writes to it
4. Import into Filament: `Texture::Builder().import((intptr_t)gl_tex_id).build(engine)`
5. Filament samples directly — zero copy

**Files**:
- MODIFY: [GaussianComputeRenderer.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeRenderer.cpp) — output texture creation

---

### PHASE 3: Metal Zero-Copy Output (macOS, parallel with Phase 2)

Eliminate readback on existing Metal backend:
1. Create MTLTexture with `usage = .shaderRead | .shaderWrite`
2. Import into Filament: `Texture::Builder().import((intptr_t)CFBridgingRetain(mtlTexture)).build(engine)`
3. Remove `DownloadMetalSharedBuffer` and `UploadOutputTextures` calls

**Files**:
- MODIFY: [GaussianComputeRenderer.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeRenderer.cpp) — Metal backend (~lines 1045-1390)

---

### PHASE 4: Depth-Based Scene Compositing

**Current approach**: The 3DGS output is rendered as an ImGui overlay on top of the Filament
scene with alpha blending ([SceneWidget.cpp](cpp/open3d/visualization/gui/SceneWidget.cpp) lines 1133-1143).
This means 3DGS always appears in front of any Filament geometry, regardless of actual depth.

**New approach**: Replace alpha-blending overlay with depth-based per-pixel compositing.
Where the Filament scene is nearer, show the scene pixel; where the 3DGS is nearer, show
the 3DGS pixel.

**Step 4.1: Create a depth-aware compositing Filament material**

New material `gs_depth_composite.mat` that samples both layers:

```
material {
    name : gsDepthComposite,
    parameters : [
        { type : sampler2d, name : gsColor },    // 3DGS color  (RGBA16F)
        { type : sampler2d, name : gsDepth },    // 3DGS depth  (R32F linear depth)
        { type : sampler2d, name : sceneDepth }, // Filament scene depth
        { type : float,     name : nearPlane },
        { type : float,     name : farPlane }
    ],
    requires : [ uv0 ],
    shadingModel : unlit,
    culling : none,
    depthCulling : false,
    depthWrite : false,
    blending : opaque
}

fragment {
    void material(inout MaterialInputs material) {
        prepareMaterial(material);
        vec2 uv = getUV0();

        // Sample 3DGS output
        vec4 gs_color = texture(materialParams_gsColor, uv);
        float gs_linear_depth = texture(materialParams_gsDepth, uv).r;
        float gs_alpha = gs_color.a;

        // Sample Filament scene depth (non-linear, need to linearize)
        float scene_ndc_depth = texture(materialParams_sceneDepth, uv).r;
        float near = materialParams.nearPlane;
        float far = materialParams.farPlane;
        float scene_linear_depth = near * far / (far - scene_ndc_depth * (far - near));

        // Depth test: choose the nearer pixel
        // If 3DGS has no content (alpha ≈ 0), always show scene
        // If scene depth is at far plane, show 3DGS if it has content
        bool gs_nearer = (gs_alpha > 0.001) && (gs_linear_depth < scene_linear_depth);
        
        if (gs_nearer) {
            // Show 3DGS pixel (convert straight-alpha to final RGB)
            material.baseColor = gs_color;
        } else {
            // Discard to show the underlying scene (rendered before this pass)
            // OR: blend the 3DGS behind scene objects
            // For partially transparent 3DGS pixels behind scene objects,
            // we might want to still show the scene pixel at full opacity.
            material.baseColor = vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
}
```

**Alternative approach**: Instead of a separate material, modify the composite compute shader
([gaussian_composite.comp](cpp/open3d/visualization/rendering/filament/shaders/gaussian_composite.comp))
to take the Filament scene depth buffer as an additional input, and incorporate it into the
front-to-back accumulation loop. This would let scene geometry participate naturally in the
depth-sorted blend, producing correct semi-transparent splats in front of AND behind scene objects.

**Step 4.2: Update gaussian_composite.comp for scene depth awareness**

Modify the composite shader to accept the Filament scene depth texture as an additional
binding. During the front-to-back splat accumulation, when a splat's depth exceeds the
scene depth at that pixel, stop accumulating splats (the scene surface occludes remaining
splats). Then blend the accumulated 3DGS color with the scene color based on the accumulated
3DGS transmittance.

New binding in composite shader:
```glsl
layout(binding = 14, r32f) uniform readonly image2D scene_depth;
```

Modified accumulation loop (conceptual):
```glsl
// During front-to-back accumulation:
float scene_depth_at_pixel = imageLoad(scene_depth, pixel).r;  // linearized
// ... for each splat in depth order:
if (splat_depth >= scene_depth_at_pixel) {
    // Scene surface is nearer — remaining splats are behind the scene
    // Mix scene color with accumulated 3DGS at current transmittance
    break;
}
// ... normal accumulation continues for splats in front of the scene
```

Output changes:
- Write premultiplied RGBA (premul alpha = 1 - remaining_transmittance)
- Or write straight-alpha with alpha = 1.0 where scene was chosen

**Step 4.3: Update SceneWidget compositing**

Replace the current two-image ImGui overlay with the depth-composited output.
Either:
- (A) A single fullscreen quad with `gs_depth_composite` material, or
- (B) The compute shader produces the final composited image directly (preferred —
  avoids an extra rendering pass)

**Files**:
- MODIFY: [gaussian_composite.comp](cpp/open3d/visualization/rendering/filament/shaders/gaussian_composite.comp) — add scene depth input, depth-aware accumulation
- NEW: `gs_depth_composite.mat` (if using material approach) in `cpp/open3d/visualization/gui/Materials/`
- MODIFY: [SceneWidget.cpp](cpp/open3d/visualization/gui/SceneWidget.cpp) — update compositing code
- MODIFY: [FilamentView.cpp](cpp/open3d/visualization/rendering/filament/FilamentView.cpp) — expose scene depth texture to compute

---

### PHASE 5: Remove Vulkan 3DGS Code

Remove all Vulkan-specific 3DGS compute code since OpenGL is now the backend for
Linux/Windows.

**Files to delete**:
- DELETE: [GaussianComputeVulkanPipeline.h](cpp/open3d/visualization/rendering/filament/GaussianComputeVulkanPipeline.h)
- DELETE: [GaussianComputeVulkanPipeline.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeVulkanPipeline.cpp)

**Files to modify**:
- MODIFY: [GaussianComputeRenderer.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeRenderer.cpp) — remove `GaussianComputeVulkanBackend` class (~400 lines), remove Vulkan includes
- MODIFY: [FilamentNativeInterop.h/.cpp](cpp/open3d/visualization/rendering/filament/FilamentNativeInterop.h) — remove `GetFilamentVulkanNativeHandles()` and `FilamentVulkanNativeHandles` struct (if only used by 3DGS)
- MODIFY: CMakeLists.txt — remove Vulkan pipeline source from build, remove Vulkan dependencies for 3DGS target

**Keep**:
- All 9 `.comp` shader source files — they are valid GLSL and compiled to OpenGL SPIR-V
- Vulkan extensions in shaders (`GL_KHR_shader_subgroup_*`) — these are Khronos standard extensions supported in OpenGL 4.6
- The `.spv` (Vulkan SPIR-V) and `.metal` outputs in the build can be removed or retained for reference

---

### PHASE 6: Build Integration

**Step 6.1: CMake changes**
- MODIFY: [Open3DAddComputeShaders.cmake](cmake/Open3DAddComputeShaders.cmake) — add OpenGL SPIR-V output; optionally remove Vulkan SPIR-V output
- MODIFY: `cpp/open3d/visualization/rendering/filament/CMakeLists.txt` — add OpenGL pipeline sources, remove Vulkan pipeline sources, link GL/EGL
- MODIFY: Top-level CMakeLists.txt — GL 4.6 capability check

**Step 6.2: Platform guards**
- OpenGL backend: `#if !defined(__APPLE__)` (Linux + Windows)
- Metal backend: `#if defined(__APPLE__)` (macOS only)
- No Vulkan backend for 3DGS

---

## File Inventory

### New files
- `GaussianComputeOpenGLPipeline.h/.cpp` — OpenGL compute API (SSBO, images, dispatch)
- `GaussianComputeOpenGLContext.h/.cpp` — EGL/WGL context creation and management
- `gs_depth_composite.mat` — Filament material for depth-based compositing (if material approach)

### Modified files
- [FilamentEngine.cpp](cpp/open3d/visualization/rendering/filament/FilamentEngine.cpp) — Linux default → kOpenGL
- [GaussianComputeRenderer.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeRenderer.cpp) — Add OpenGL backend, remove Vulkan backend, update Metal for import(), update CreateBackend/Supported
- [gaussian_composite.comp](cpp/open3d/visualization/rendering/filament/shaders/gaussian_composite.comp) — Add scene depth input for depth-based compositing
- [gaussian_radix_sort*.comp](cpp/open3d/visualization/rendering/filament/shaders/) (4 files) — push_constant → uniform block
- [Open3DAddComputeShaders.cmake](cmake/Open3DAddComputeShaders.cmake) — OpenGL SPIR-V compilation
- [SceneWidget.cpp](cpp/open3d/visualization/gui/SceneWidget.cpp) — Depth-based compositing
- [FilamentView.cpp](cpp/open3d/visualization/rendering/filament/FilamentView.cpp) — Expose scene depth to compute pipeline
- [FilamentNativeInterop.h/.cpp](cpp/open3d/visualization/rendering/filament/FilamentNativeInterop.h) — Remove Vulkan interop, add OpenGL if needed

### Deleted files
- [GaussianComputeVulkanPipeline.h](cpp/open3d/visualization/rendering/filament/GaussianComputeVulkanPipeline.h)
- [GaussianComputeVulkanPipeline.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeVulkanPipeline.cpp)

### Reference files (read-only)
- [GaussianComputeDataPacking.h/.cpp](cpp/open3d/visualization/rendering/filament/GaussianComputeDataPacking.h) — scene packing (unchanged)
- [GaussianComputeMetalShaders.h/.mm](cpp/open3d/visualization/rendering/filament/GaussianComputeMetalShaders.h) — Metal compute (modified for import only)
- [Texture.h](build/filament/src/ext_filament/include/filament/Texture.h) — import() API
- [PlatformEGL.h](build/filament/src/ext_filament/include/backend/platforms/PlatformEGL.h) — shared context

---

## Dependency Graph

```
Phase 1 (Linux → OpenGL default) ──── independent, do first
Phase 2 steps:
  Step 2.1 (shared GL context) ────────┐
  Step 2.3 (shader SPIR-V) ───────────┤
                                       ├─→ Step 2.4 (OpenGL backend)
  Step 2.2 (OpenGL pipeline API) ──────┤     └─→ Step 2.5 (zero-copy)
                                       │
Phase 3 (Metal zero-copy) ──── parallel with Phase 2
Phase 4 (depth compositing) ── depends on Phase 2 (needs scene depth access)
Phase 5 (remove Vulkan) ────── depends on Phase 2 (replacement must work first)
Phase 6 (build integration) ── depends on Phase 2, 5
```

---

## Verification

1. **Build**: `cmake --build build --parallel $(nproc) --target GaussianSplat` on Linux + Windows
2. **Runtime**: `./bin/examples/GaussianSplat <scene>.ply`
   - Scene renders correctly; camera rotation/zoom works; no flicker on mouse-move
3. **Wayland**: Test on Wayland session (e.g. `GDK_BACKEND=wayland`) — verify window creation and rendering
4. **Depth compositing**: Place Filament geometry (cube, mesh) partially inside Gaussian splat scene,
   verify correct occlusion in both directions (scene in front of splats AND splats in front of scene)
5. **GL debug**: Enable `GL_KHR_debug` for compute dispatch error checking
6. **Metal**: Verify zero-copy on macOS via Xcode GPU profiler
7. **Performance**: Frame time comparison before/after readback elimination (~2-4ms expected improvement)

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Linux default → OpenGL | Ensures GS compute works on all Linux systems; Vulkan was the previous default but not needed for 3DGS |
| EGL for shared context (not GLX) | EGL works on both X11 and Wayland; GLX is X11-only |
| Remove Vulkan 3DGS code | Simplifies maintenance; OpenGL compute is sufficient for 3DGS workload; Vulkan import() doesn't work in Filament |
| Keep shader Vulkan extensions | `GL_KHR_shader_subgroup_*` are Khronos standard, supported in OpenGL 4.6 |
| Build-time SPIR-V (not runtime GLSL) | Consistency with existing pipeline; faster startup; reproducible builds |
| Vulkan SPIR-V in OpenGL (not OpenGL SPIR-V) | OpenGL SPIR-V (`-G`) fails for subgroup ops; Vulkan SPIR-V (`-V`) works via `GL_ARB_gl_spirv` |
| GL 4.6 minimum for compute | Required for `glShaderBinary(SPIR_V)` + `glSpecializeShader()`. Widely available (Mesa 18.0+, 2017+) |
| Depth-based compositing | Enables mixed 3DGS + mesh scenes with correct mutual occlusion |
| import() for zero-copy | Works on OpenGL and Metal; eliminates ~12 MB/frame PCIe traffic |
