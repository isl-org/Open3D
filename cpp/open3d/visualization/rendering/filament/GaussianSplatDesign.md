# 3D Gaussian Splatting Rendering in Open3D — Design & Implementation

## Overview

Open3D supports real-time 3D Gaussian Splatting (3DGS) rendering through a GPU compute pipeline
that runs alongside the Filament-based visualization engine. The compute pipeline projects,
sorts, and composites Gaussian splats into a color image. A shared GL depth texture allows
the composite shader to reject splats behind Filament-rendered mesh geometry, producing
correct per-splat depth occlusion.

**Supported platforms**: Linux X11/GLX (operational, including Wayland via XWayland),
Windows/WGL (SPIR-V shader loading error, fix pending), macOS/Metal (operational).

---

## Algorithm: 3D Gaussian Splatting (3DGS) Rendering

To implement a **3D Gaussian Splatting (3DGS)** renderer from scratch, you must handle specific geometric transformations and high-performance CUDA-style rasterization. Below are the granular details required for an implementation.

---

### 1. The Gaussian Primitive Representation
Each Gaussian $i$ is stored in a structured buffer with the following parameters:

* **Position ($\mu_i$):** $3 \times 1$ vector.
* **Rotation ($q_i$):** A normalized unit quaternion $[w, x, y, z]$.
* **Scale ($s_i$):** A $3 \times 1$ vector $[s_x, s_y, s_z]$ (stored in log-space, exponentiated for rendering).
* **Opacity ($\alpha_i$):** A scalar, transformed via sigmoid $\alpha = \frac{1}{1 + e^{-x}}$ to bound it in $[0, 1]$.
* **Spherical Harmonics ($SH_i$):** Up to 16 coefficients per color channel (for Degree 3). Total of 48 values ($16 \times 3$).

The 3D covariance matrix $\Sigma$ is reconstructed as:
$$\Sigma = R S S^T R^T$$
Where $R$ is the rotation matrix derived from $q_i$, and $S$ is the diagonal scaling matrix.

---

### 2. Projection (The "Splat" Math)
To project a 3D Gaussian to 2D, you need the viewing transformation $W$ and the camera projection matrix $P$.

1.  **View Space Mean:** $x_{view} = W \mu_i$.
2.  **2D Covariance ($\Sigma'$):** Using the Jacobian $J$ of the perspective projection:
    $$\Sigma' = J W \Sigma W^T J^T$$
    * **Low-pass Filter:** A small identity matrix ($0.3 \times I_{2 \times 2}$) is added to $\Sigma'$ to ensure the splat covers at least one pixel (anti-aliasing).
3.  **Eigenvalue Decomposition:** Calculate the eigenvalues of $\Sigma'$ to determine the 2D radius. Cull if the center is outside the frustum or the radius is negligible.

---

### 3. Tile-Based Sorting & Culling
This is the core of the 3DGS performance.

* **Grid:** Divide the $W \times H$ screen into tiles of $16 \times 16$ pixels. Choose  number of bits for tile ID as $T = ceil(\log_2 (W / 16 * H/16))$
* **Duplicate and Key:** If a Gaussian's 2D radius overlaps $N$ tiles, create $N$ instances of a 32-bit key:
    * **High T bits:** Tile ID (row-major order).
    * **Low 32-T bits:** Depth (distance from camera) without sign bit.
* **Radix Sort:** Sort these keys to group Gaussians by tile and depth.
* **Ranges:** Identify the `start` and `end` index in the sorted list for each Tile ID.

---

### 4. The Rasterization Kernel
For each tile (one CUDA block), execute the following for every pixel $(x, y)$ in parallel:

#### A. Fetching Data
Load Gaussian data into **Shared Memory** in chunks (e.g., blocks of 256) to minimize global memory latency.

#### B. Influence Calculation
For a pixel at $(x, y)$, calculate the offset to the Gaussian mean $d = [x - u_i, y - v_i]$. The contribution $G_i$ is:
$$G_i(x, y) = \exp \left( -\frac{1}{2} d^T (\Sigma'_i)^{-1} d \right)$$

#### C. Alpha Blending (Front-to-Back)
Accumulate color $C$ and transmittance $T$:
1.  **Effective alpha:** $\alpha_{eff} = \alpha_i \cdot G_i(x, y)$.
2.  **Update Color:** $C_{final} = C_{final} + C_i \cdot (\alpha_{eff} \cdot T)$.
3.  **Update Transmittance:** $T = T \cdot (1 - \alpha_{eff})$.
4.  **Early Exit:** If $T < 0.0001$, stop processing for that pixel.

---

### 5. Specific Parameter Values

| Parameter | Typical Value | Purpose |
| :--- | :--- | :--- |
| **Tile Size** | $16 \times 16$ | Optimized for GPU warp architecture. |
| **Early Exit $\epsilon$** | $1/255$ | Stops processing once a pixel is saturated. |
| **SH Degree** | 3 | High-frequency view-dependent reflections. |
| **Filter Size** | $0.3$ px | Prevents aliasing on sub-pixel Gaussians. |
| **Culling Margin** | 3.0 | Gaussians rendered up to $3\sigma$ from mean. |

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
window when context sharing can be established — GLX/WGL sharing is set at context creation
time and cannot be added retroactively.

1. `FilamentEngine.cpp` calls `GaussianSplatOpenGLContext::GetInstance().InitializeStandalone()`
2. `InitializeStandalone()` creates a hidden GLFW OpenGL 4.6 helper window
3. The resulting native context handle is passed to `Engine::create()` as `sharedGLContext`
4. Filament's OpenGL platform creates its own context sharing with that native handle
  (`PlatformGLX` on Linux/X11/XWayland, `PlatformWGL` on Windows)
5. Both contexts now share the same GL object namespace — `import()` works by handle

There is no supported late-initialization fallback anymore. If the helper context is not
created before `Engine::create()`, zero-copy sharing cannot be established retroactively.

### Backend Abstraction

```
GaussianSplatRenderer::Backend (abstract)
├── GaussianSplatOpenGLBackend      — Linux + Windows (GL 4.6 core compute, SPIR-V)
├── GaussianSplatMetalBackend       — macOS (Metal compute, operational)
└── GaussianSplatPlaceholderBackend — fallback (logs once per view, returns false)
```

Each backend implements `RenderGeometryStage` (passes 1-4) and `RenderCompositeStage`
(pass 5). The split enables GPU overlap: Stage A dispatches compute work without `glFinish()`,
allowing Filament rasterization to execute concurrently. Stage B waits for Filament to
complete (`flushAndWait`) before reading scene depth.

---

## Sort Key Layout

Each TileEntry sort key packs `tile_index` and `depth` into a single 32-bit uint using a
dynamic split computed from the actual tile count:

```
bits 31..D   tile_index   (T bits, T = ceil(log2(tile_count)), clamped to [1,31])
bits D-1..0  depth field  (D = 32-T bits of depth precision)
```

`key = (tile_index << D) | ((depth_key << 1u) >> T)`

The `<< 1` strips the IEEE 754 sign bit from `depth_key`, which is always 0 because depth
is clamped to `>= 0` before `floatBitsToUint()` in the scatter pass.  This reclaims one free
depth bit at no cost.

T is computed CPU-side as `floor(log2(max_tile_index)) + 1` and stored in
`GaussianViewParams.limits.w`, then read in `gaussian_radix_sort_keygen.comp`.

| Viewport | Tiles (16×16) | T (tile bits) | D (depth bits) |
|----------|---------------|---------------|----------------|
| 1080p    | 8,160         | 13            | 19             |
| 4K       | 32,400        | 15            | 17             |
| 8K       | 129,600       | 17            | 15             |

The fixed 16/16 split used previously wasted the sign bit and gave 16 depth bits at all
resolutions.

The 4-pass radix sort operates on all 32 bits (8 bits per pass), so the key width change
requires no dispatch logic updates.

---

## Completed Work

### PHASE 1: Linux/Windows Default → OpenGL (DONE)
- `FilamentEngine.cpp`: OpenGL is the default for both Linux and Windows.
- macOS uses the default Filament backend (Metal).

### PHASE 2: OpenGL Compute Backend (DONE)
- **GL context** (`GaussianSplatOpenGLContext`): Standalone GL 4.6 core-profile context created before Filament,
  passed as `sharedGLContext` to `Engine::create()`. Shares GL namespace with Filament's context.
  Created via GLFW: GLX on Linux X11/XWayland, WGL on Windows. Linux offscreen rendering requires
  an X11 or XWayland server.
- **Pipeline API** (`GaussianSplatOpenGLPipeline`): Thin wrappers around GL 4.6 core compute:
  SSBO/UBO bind, image/sampler bind, buffer management, texture management, dispatch, barriers.
  All GL constants as `constexpr` to avoid GL header dependency in consumers.
- **SPIR-V loading**: 9 shaders compiled offline from GLSL `.comp` to Vulkan-targeted SPIR-V
  (`.spv`); loaded via `glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V)` + `glSpecializeShader()`.
  OpenGL SPIR-V target (`-G`) fails for subgroup ops; Vulkan target works via `GL_ARB_gl_spirv`.
- **Radix sort**: 4-pass 8-bit LSD radix sort. Runtime-queries `gl_SubgroupSize`. Sort key:
  `(tile_index << D) | ((depth_key << 1) >> T)` where T = `limits.w` = ceil(log2(tile_count))
  (dynamic split, sign-bit stripped from depth). Keygen reads `view_params` at binding 0.
- **Two-stage execution**: `RenderGeometryStage` (passes 1-4) defers `glFinish()` to enable
  overlap. `RenderCompositeStage` (pass 5) runs after `flushAndWait()`.
- **Zero-copy output**: `PrepareOutputTargets` creates GL textures (`DEPTH_COMPONENT32F` +
  `RGBA16F`), imports into Filament via `Texture::Builder::import()`. No CPU staging.
- **Depth-aware compositing**: Composite shader reads scene depth at binding 14, linearizes
  Filament's reversed-Z depth, discards splats behind mesh geometry per-pixel.
- **MSAA disabled** for GS views (required for sampleable depth attachment).
- **Counter readback**: After prefix-sum pass, `total_entries` is read back from a GPU counter
  buffer to size tile entries and sort buffers before scatter.

### PHASE 3: macOS / Metal Backend (DONE)
- **Metal GPU context** (`GaussianSplatGpuContextMetal`): Acquires Filament's `MTLDevice` and
  `MTLCommandQueue` via `FilamentNativeInterop` (`GetFilamentMetalNativeHandles`). Loads all 9
  compute pipelines from a pre-built `.metallib` (compiled by CMake via SPIRV-Cross from the same
  GLSL `.comp` sources). All dispatch, barrier, and buffer-binding operations fully implemented.
- **Pass execution** (`GaussianSplatMetalBackend`): Calls `RunGaussianGeometryPasses` and
  `RunGaussianCompositePass` from the shared `GaussianSplatPassRunner` — identical pass logic
  to the OpenGL backend.
- **Metal output targets** (`GaussianSplatOutputTargetsApple`): Creates `MTLTexture` objects
  (`Depth32Float` + `RGBA16Float`), imports into Filament via `CreateImportedMTLTexture()`.
  MSAA disabled for GS views (sampleable depth incompatible with MSAA, same as GL).
- **Frame schedule**: Geometry stage dispatches during `BeginFrame()` on a dedicated geometry
  `MTLCommandBuffer`. Composite stage runs after `renderer_->endFrame()` in `EndFrame()`,
  guaranteeing the depth texture is fully produced before the composite samples it.
- Validated on Apple Silicon (M-series) hardware.

### PHASE 4: Depth-Based Scene Compositing (DONE)
- `gaussian_composite.comp`: binding 14 = `sampler2D scene_depth`. Reversed-Z linearization.
  Per-splat occlusion test: `if (use_scene_depth && s_depth[i] >= scene_linear_depth) continue`.
- UBO `depth_range_and_flags[3]` set to `1.0` in `RenderCompositeStage` when scene depth valid.
- Example (`GaussianSplat.cpp`): red sphere placed at scene bbox center for depth compositing
  testing.

### PHASE 5: Vulkan Code Removal (DONE)
- `GaussianSplatVulkanPipeline.h/.cpp` deleted.
- `GaussianSplatVulkanBackend` class removed from `GaussianSplatRenderer.cpp`.

### Bug Fixes (DONE)
- **Tile index overflow** (`gaussian_radix_sort_keygen.comp`): key was `tile_index << 20`
  (12-bit tile field, max 4095 tiles). Changed to `tile_index << 16` (16 bits, max 65535).
  Fixed corruption on windows larger than ~1280×720.
- **Use-after-free on resize**: `FilamentView::EnableViewCaching()` freed `color_buffer_`
  while the GS render target still referenced it. Fixed by calling
  `scene_->InvalidateGaussianSplatOutput(*this)` before freeing `color_buffer_`, tearing
  down the GS render target first through:
  `FilamentScene → FilamentRenderer → GaussianSplatRenderer::InvalidateOutputForView()`.
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
| Opt11 | No-readback radix sort | `gaussian_compute_dispatch_args.comp` writes all indirect dispatch counts and `RadixSortParams` GPU-side after prefix-sum; removes `DownloadGLBuffer` CPU stall (FW1-1). Sort buffers pre-allocated from splat-count estimate; scatter dispatched over `splat_count` for full parallelism. |
| Opt12 | Async Stage A overlap | Stage A dispatches geometry + sort compute without `glFinish()`; `renderer_->beginFrame()` fires immediately after, so Stage A GPU work runs concurrently with Filament's rasterization of the same frame. On Metal, geometry and composite use separate `MTLCommandBuffer` objects that the GPU schedules independently (FW1-4). |
| Opt13 | Compressed input buffers | `log_scales` and `dc_opacity` stored as fp16 (8 B/splat each, `uvec2` per splat); `rotations` as snorm8-biased uint (4 B/splat); `sh_coefficients` fp16 and degree-dependent (0/24/48 B/splat for degree 0/1/2). Total input: **36–84 B/splat** vs. the previous 160 B. GPU decodes via `unpackHalf2x16` and a 3-instruction `DecodeSnorm8` (core GLSL 4.2+, no extension). |

### GPU Object Labeling

All GPU objects are labeled at construction time for debugger visibility (RenderDoc, Metal
Frame Debugger, etc.):

**GL programs** (`glObjectLabel(GL_PROGRAM, ...)`):
Labeled with the shader source filename (e.g. `gaussian_project.comp`, `gaussian_composite.comp`).
Applied inside `LoadGLComputeProgramSPIRV` and `LoadGLComputeProgramGLSL` immediately after
a successful `glLinkProgram`.

**GL buffers** (`glObjectLabel(GL_BUFFER, ...)`):
All 20 per-view SSBOs/UBOs are labeled using the `gs.*` naming scheme
(e.g. `gs.projected`, `gs.tile_entries`, `gs.sorted_entries`).  Labels are applied at
`ResizeBuffer`/`ResizePrivateBuffer` time, including the reuse path when an existing buffer
is large enough.

**GL textures** (`glObjectLabel(GL_TEXTURE, ...)`):
- `gs.scene_depth` — `GL_DEPTH_COMPONENT32F`; Filament writes, composite reads.
- `gs.color` — `GL_RGBA16F`; composite writes, ImGui samples.
- `gs.composite_depth` — `GL_R32F`; composite writes per-splat depth.
Labels are applied via `CreateGLTexture2D` / `ResizeGLTexture2D`.

**Metal objects** (`setLabel:`):
- Buffers: same `gs.*` scheme as GL, applied in `AllocateBuffer` and reuse paths.
- Textures: `gs.scene_depth`, `gs.color` in `GaussianSplatOutputTargetsApple`;
  `gs.composite_depth` in `ComputeGPUMetal`'s `CreateTexture2DR32F`.

**CPU/GPU struct name alignment**:
CPU C++ struct names match their GLSL counterparts exactly so debugger displays
are unambiguous:

| C++ struct | GLSL name | Location |
|---|---|---|
| `GaussianViewParams` | `GaussianViewParams` | UBO binding 0 |
| `ProjectedGaussian` | `ProjectedGaussian` | SSBO binding 6 |
| `TileEntry` | `TileEntry` | SSBOs binding 7/8/9 |
| `RadixSortParams` | `RadixSortParams` | UBO binding 14 |

---



The projection pass always adds a subpixel blur (+0.3 on the diagonal of the projected 2D covariance) to regularise very small splats.  Without further correction this makes tiny splats appear artificially bright by widening their effective footprint.

When `RenderConfig::antialias = true`, the projection shader applies a **density compensation factor** that cancels the brightness increase:

$$\text{compensation} = \sqrt{\frac{\det(\Sigma_{\text{orig}})}{\det(\Sigma_{\text{blurred}})}}$$

The final opacity becomes `alpha *= compensation`.  The ratio is clamped to `max(0, …)` before the square root so degenerate (zero-area) splats are silently zeroed out rather than producing NaN.

This mirrors [gsplat PR #117](https://github.com/nerfstudio-project/gsplat/pull/117).  The flag is transmitted via `scene.z` of the per-frame UBO (`GaussianViewParams`), so toggling it costs nothing more than a 4-byte UBO update per frame.

### Data Packing (`GaussianSplatDataPacking`)
- `PackGaussianViewParams`: packs view/projection matrices + scene scalars into the 288-byte
  `GaussianViewParams` UBO. Called every frame; no heap allocation.
- `PackGaussianSceneAttributes`: packs per-splat geometry into compressed GPU layouts.
  Called only when the scene changes.
- `GaussianViewParams` (std140 UBO, binding 0): matrices, viewport, scene params.
  `scene.z` = antialias flag (0 = off, 1 = density-compensation on).
  `limits.x` = tile-entry capacity actually allocated for the frame.
  `limits.y` = `RenderConfig::max_tiles_per_splat`.
  `limits.z` = `RenderConfig::max_tile_entries_total`.
  `depth_range_and_flags.w` = scene depth flag used by composite (0.0 = no, 1.0 = yes).
- `ProjectedGaussian`: 48-byte per-splat descriptor (output of projection pass)
- `TileEntry`: 16-byte per tile-entry sort record (`depth_key`, `splat_index`,
  `stable_index` — currently equals `splat_index`, reserved for future stable-sort support,
  `tile_index` — linear tile index written by scatter and read by keygen)

### Runtime Capacity Limits And Error Flags

`RenderConfig` now exposes two runtime safety knobs used by both CPU allocation
and GPU-side clamping:

- `max_tiles_per_splat` — per-splat budgeting term used to estimate the tile-entry buffer size.
- `max_tile_entries_total` — hard ceiling for total tile entries, sort keys, and sort values.

The packed `GaussianViewParams.limits` block carries those values to the shaders so
all stages use the same capacity value.

`counters_buf` is also used as a compact GPU→CPU diagnostics channel,
mapped to the `GaussianGpuCounters` struct in `GaussianSplatDataPacking.h`
(GLSL binding 10, `gs_counters`):

| Field | Meaning |
|---|---|
| `total_entries` | Raw total tile entries written by prefix-sum |
| `error_flags` | Error bitmask (see below) |
| `tile_count` | Total tile count for the frame |
| `splat_count` | Visible splat count for the frame |

Current error bits:

- Bit 0: tile-entry overflow in `gaussian_scatter.comp`; excess entries are dropped.
- Bit 1: dispatch/sort count clamped in `gaussian_compute_dispatch_args.comp` because raw total entries exceeded the configured capacity.

The pass runner downloads this small bitmask after GPU work completes and logs each
warning once per view. This keeps the steady-state path cheap while still surfacing
capacity-related rendering degradation to users.

**Per-splat input SSBO formats** (consumed by `gaussian_project.comp`):

| Buffer | Binding | GPU type | B/splat | CPU packing |
|--------|---------|----------|---------|-------------|
| `positions` | 1 | `vec4` fp32 | 16 | direct copy |
| `log_scales` | 2 | `uvec2` fp16×4 | 8 | `PackHalf2(·,·)` ×2 |
| `rotations` | 3 | `uint` snorm8-biased×4 | 4 | `PackSnorm8x4(w,x,y,z)` |
| `dc_opacity` | 4 | `uvec2` fp16×4 | 8 | `PackHalf2(·,·)` ×2 |
| `sh_coefficients` | 5 | `uvec2` fp16, stride=3×degree | 0/24/48 | `PackHalf2` pairs for `(((degree + 1)^2 - 1) * 3)` rest coefficients (`f_rest`; DC lives in `dc_opacity`) |
| **Total** | | | **36/60/84** | (was 160 B for all degrees) |

---

## Shader Variant System

### Problem

On Windows OpenGL (confirmed on Intel GPU), `GL_KHR_shader_subgroup_arithmetic`
is miscompiled both when loading Vulkan-targeted SPIR-V via `GL_ARB_gl_spirv`
and when using the online GLSL compilation path. The scatter pass produces
`ValuesOut` entries larger than `g_num_elements`, causing corrupt rendering.
The same shaders work correctly on Linux (GLX) and macOS (Metal/MSL).

Only two of the nine Gaussian compute shaders are affected — those that use
`subgroupAdd`, `subgroupExclusiveAdd`, and `subgroupElect`:

| Shader | Subgroup ops |
|---|---|
| `gaussian_prefix_sum` | `subgroupAdd`, `subgroupExclusiveAdd`, `subgroupElect` |
| `gaussian_radix_sort` | same |

### Variants

Each affected shader now has two files:

| File | Subgroup ops | Recommended platform |
|---|---|---|
| `gaussian_prefix_sum.comp` / `.spv` | None (thread-0 scan) | Windows + unknown GPUs |
| `gaussian_prefix_sum_subgroup.comp` / `.spv` | Yes | Linux, macOS |
| `gaussian_radix_sort.comp` / `.spv` | None (thread-0 scan) | Windows + unknown GPUs |
| `gaussian_radix_sort_subgroup.comp` / `.spv` | Yes | Linux, macOS |

All other seven shaders have a single file unchanged by this change.

### Runtime Policy

The shader variant and loader are controlled by two fields on
`GaussianSplatRenderer::RenderConfig`:

| Field | Type | Default (Windows) | Default (Linux/macOS) |
|---|---|---|---|
| `use_shader_subgroups` | `bool` | `false` | `true` |
| `use_precompiled_shaders` | `bool` | `false` | `true` |

These fields are populated at `GaussianSplatRenderer` construction time from
the `RenderConfig` defaults (platform-selected via `#ifdef _WIN32`).  They are
passed to `CreateComputeGpuContextGL(use_subgroups, use_precompiled)` and stored
in the GL context object.

**Load sequence for each shader** (`EnsureProgramsLoaded`, `LoadOneProgram`):

```
resolve file_base:
  if shader name ends with "_subgroup" AND !use_subgroups:
    strip "_subgroup" suffix  → load portable no-subgroup variant
  else:
    use name as-is

if use_precompiled:
  try load file_base.spv via GL_ARB_gl_spirv
  → success: done
  → failure: log warning, fall through to GLSL

load file_base.comp via online GLSL compilation
```

The `kGsShaderNames` table lists all shader base names with `_subgroup` suffix
for the two affected shaders.  The suffix-stripping logic in `LoadOneProgram`
makes it unnecessary to maintain a separate `kSubgroupCapableShaders` set.

**Auto-fallback:** If *any* shader fails to load under the primary policy, the
entire set is discarded and all programs are retried with the safe fallback
`{use_subgroups=false, use_precompiled=false}`. This ensures a working render
on uncharted GPU/driver combinations without user intervention. A warning is
logged when fallback activates.
is logged when fallback activates.

### MaterialRecord / Python API

Two capacity limit fields exposed on `MaterialRecord` (and via Python bindings)
allow per-scene tuning without touching C++ headers:

| `MaterialRecord` field | `RenderConfig` field | Default |
|---|---|---|
| `gaussian_splat_max_tiles_per_splat` | `max_tiles_per_splat` | 32 |
| `gaussian_splat_max_tile_entries_total` | `max_tile_entries_total` | 32 × 1024 × 1024 |

`FilamentScene::CacheGaussianSplatData` propagates these values to the renderer's
`RenderConfig` via `SetRenderConfig` whenever splat data is loaded.

The shader-loading policy fields (`use_shader_subgroups`,
`use_precompiled_shaders`) are not surfaced to `MaterialRecord` because they
must be set before the GPU context is constructed (i.e., at renderer creation
time). They can be modified via `GaussianSplatRenderer::SetRenderConfig` only
before the first render call.

### Build

`Open3DAddComputeShaders.cmake` processes all eleven `.comp` files in
`GAUSSIAN_COMPUTE_SHADER_SOURCE_FILES`:

- Each file is staged as-is (`.comp`) for online GLSL fallback.
- Each file is compiled to `.spv` (Vulkan 1.3 target) via `glslangValidator`.
- On Apple, `.spv` is transpiled to `.metal` via `spirv-cross`, with
  `--msl-fixed-subgroup-size 32` applied to the two `_subgroup` variants.
  The non-subgroup variants do not use `gl_SubgroupSize` and do not need the flag.

```
gaussian_compute/
  gaussian_project.comp / .spv
  gaussian_prefix_sum.comp / .spv          ← no-subgroup (Windows default)
  gaussian_prefix_sum_subgroup.comp / .spv ← subgroup    (Linux/macOS default)
  gaussian_radix_sort.comp / .spv          ← no-subgroup
  gaussian_radix_sort_subgroup.comp / .spv ← subgroup
  … (7 other shaders, one variant each)
  gaussian_compute.metallib               ← Apple only
```

---

## Planned Work

### PHASE 3: macOS / Metal Backend (DONE)

See Completed Work → PHASE 3 for implementation details.

### PHASE 6: Build Integration Cleanup

- ~~Remove `BUILD_GAUSSIAN_SPLAT_COMPUTE` CMake option~~ — **Done.** Gaussian
  splatting is now always built when `BUILD_GUI=ON`.
- ~~Remove `OPEN3D_SHADER_SUBGROUPS`/`OPEN3D_SHADER_PRECOMPILED` env var overrides~~
  — **Done.** Policy is now sourced from `RenderConfig` with platform defaults.
- ~~Remove `OPEN3D_GAUSSIAN_DEBUG_SORT` sort-debug env var and all associated
  dump functions~~ — **Done.** Development-only debug code was deleted.
- Remove stale `resources/gaussian_compute/` copies (now generated from `shaders/` by CMake)
- Add GL 4.5 minimum capability check to CMake (`GL_ARB_compute_shader`,
  `GL_ARB_shader_storage_buffer_object`)
- Verify SPIR-V shader compilation across all supported glslangValidator versions
- ~~Fix SPIR-V shader loading on Windows~~ — resolved via no-subgroup + online GLSL
  default on Windows; see Shader Variant System section.

---

## Future Work

### FW1: Performance Improvements

| # | Optimization | Impact | Description |
|---|---|---|---|
| FW1-1 | ~~Eliminate counter readback~~ | ~~High~~ | **Done (Opt11).** Removed `DownloadGLBuffer` stall. `gaussian_compute_dispatch_args.comp` writes all indirect dispatch args and `RadixSortParams` GPU-side. Sort buffers pre-allocated; scatter dispatched over `splat_count`. |
| FW1-2 | Reduce radix sort passes | Medium | Key is 32 bits but only `ceil(log2(tile_count)) + 16` bits are significant. At 4K (32,400 tiles), only 30 bits matter. Running fewer sort passes can reduce one full sort phase per frame. |
| FW1-3 | Fused prefix-sum + scatter | Medium | Merge tile count accumulation and scatter using subgroup prefix sums plus global atomics. Eliminates one compute dispatch and one full barrier round-trip. |
| FW1-4 | ~~Async compute overlap~~ | ~~Medium~~ | **Done (Opt12).** Stage A dispatches without `glFinish()`; Filament's `beginFrame()` follows immediately so both execute concurrently on GPU. Metal uses separate `MTLCommandBuffer` objects per stage for hardware-level scheduling overlap. |
| FW1-5 | SH degree LOD | Low–Medium | Dynamically reduce SH degree for distant/small splats to reduce bandwidth and projection cost. |
| FW1-6 | Per-tile entry budget | Low | Cap tiles-per-splat to bound worst-case sort/composite cost for huge splats. |
| FW1-7 | Indirect dispatch for scatter | Low | Use GPU-written dispatch arguments (`glDispatchComputeIndirect`) to reduce overhead on sparse scenes. |
| FW1-8 | MSAA re-enable via depth resolve | Low | Re-enable MSAA for meshes via explicit depth resolve while keeping sampleable depth for GS composite. |

### FW2: Native Wayland / EGL — Unplanned

**Status**: Unplanned. Filament does not support EGL/Wayland zero-copy texture sharing.

The depth-aware composite requires the GS compute context and Filament to share depth and
color textures without CPU copies. On Linux this relies on GL object namespace sharing
between two GLX contexts. An EGL context cannot participate in a GLX sharing group; passing
an EGL context as `sharedGLContext` to Filament's `PlatformGLX` causes a `glXQueryContext`
X11 error. There is no windowed EGL platform in Filament v1.54.0 with cross-context object
sharing support.

**Workaround**: Wayland sessions work transparently via XWayland. `GLFW_PLATFORM_X11` is
forced in `GLFWWindowSystem::Initialize()` so GLFW and Filament both use GLX on all Linux
sessions, providing full GS functionality under any Wayland compositor with XWayland enabled.

### FW3: Windows / WGL — SPIR-V Shader Loading Error

**Status**: Implementation complete; blocked on a SPIR-V shader loading error at runtime.

`GaussianSplatOpenGLContext` creates a shared GL 4.6 core-profile WGL context before Filament via
`InitializeStandalone()` (hidden 1×1 GLFW helper window).
Context sharing and zero-copy texture import use the same mechanism as GLX. However,
`gaussian_composite.comp` (and possibly other shaders) fail to load on Windows OpenGL
drivers with a SPIR-V specialization error.

**Likely cause**: The shaders are compiled with a Vulkan SPIR-V target and use subgroup
extensions (`GL_KHR_shader_subgroup`, `gl_SubgroupSize`). Some Windows drivers require
explicit extension enables in the SPIR-V binary or do not expose these extensions via
`GL_ARB_gl_spirv`.

**Required fix**: Reproduce and triage on Windows 10/11 with Intel/NVIDIA/AMD.
Options: add explicit extension-enable decorations to the SPIR-V; provide a GLSL source
fallback path on Windows; or rewrite the affected passes to avoid subgroup ops.

**Files affected**: `GaussianSplatGpuContextGL.cpp`, `shaders/gaussian_composite.comp`
(and possibly other shaders), `GaussianSplatOpenGLContext.h/.cpp`

### FW4: Native Vulkan Backend — Unplanned

**Status**: Unplanned. Filament does not support Vulkan zero-copy texture sharing.

The depth-aware composite requires GS compute and Filament to share depth and color textures
without CPU copies. Filament's Vulkan backend does not expose an API to import
externally-allocated `VkImage` / `VkDeviceMemory` objects into the renderer's image layout
tracking, making zero-copy depth import infeasible without significant changes to Filament
internals.

OpenGL (Linux/Windows via GLX/WGL) and Metal (macOS) both provide cross-context zero-copy
texture sharing — GL object namespace sharing and `MTLTexture` bridging respectively —
which is why those are the chosen backends.

---

## Known Issues / Risks

| # | Issue | Severity | Mitigation |
|---|---|---|---|
| 1 | Counter readback CPU stall | Medium | Remove readback by pre-allocation / indirect dispatch (FW1-1/FW1-7). |
| 2 | Reversed-Z assumption | Medium | Add runtime diagnostics and shader path toggle for non-reversed depth conventions. |
| 3 | Post-processing depth loss | Low–Medium | Keep `SetPostProcessing(false)` for GS views; warn when re-enabled. |
| 4 | Stage A overlap sync | Low | If driver-specific glitches appear, insert `glFlush()` at end of Stage A. |
| 5 | Redundant depth buffer on resize | Low | Avoid creating `depth_buffer_` in cached view when GS shared depth is active. |
| 6 | Native Wayland (no XWayland) unsupported | Low | Unplanned: Filament does not support EGL/Wayland zero-copy texture sharing. XWayland provides full GS functionality on Wayland compositors. See FW2. |
| 7 | Windows SPIR-V shader loading error | Medium | `gaussian_composite.comp` fails with a SPIR-V specialization error on Windows OpenGL drivers. Likely subgroup extension issue. See FW3 for details and fix options. |

---

## File Inventory

### Core implementation files
- `GaussianSplatRenderer.h/.cpp` — Backend interface, OpenGL/Metal backends, output lifecycle;
  `InvalidateOutputForView()` for safe resize
- `GaussianSplatOpenGLPipeline.h/.cpp` — GL 4.6 compute API wrappers
- `ComputeGPU.h` — All generic GPU compute types in one header:
  `ComputeProgramId` enum, `ImageFormat` enum, `GaussianSplatGpuContext` abstract base,
  `GpuComputeFrame` RAII (Begin/EndGeometryPass or Begin/EndCompositePass),
  `GpuComputePass` RAII builder (UseProgram + PushDebugGroup on ctor, Dispatch/DispatchIndirect,
  PopDebugGroup on dtor, no-op on load failure). Factory declarations included.
- `ComputeGPUGL.cpp` — OpenGL 4.6 + SPIR-V implementation of `GaussianSplatGpuContext`
- `ComputeGPUMetal.mm` — Metal implementation: buffer management, pipeline selection,
  `Dispatch()`, `DispatchIndirect()`, barrier, texture ops
- `GaussianSplatMetalBackend.mm` — Metal backend: acquires Filament `MTLDevice`/queue,
  runs geometry + composite stages
- `GaussianSplatOutputTargetsApple.h/.mm` — Creates `MTLTexture` (depth + color),
  imports into Filament via `CreateImportedMTLTexture()`
- `GaussianSplatPassRunner.h/.cpp` — Backend-agnostic geometry + composite pass sequence
  (shared by GL and Metal); each dispatch is one `GpuComputePass(ctx, id, label).SSBO().Dispatch()`
  expression; `GpuComputeFrame` ensures Begin/End pairs are always matched
- `FilamentNativeInterop.h/.mm` — Retrieves Filament `MTLDevice` and `MTLCommandQueue`
  from `PlatformMetal`
- `GaussianSplatOpenGLContext.h/.cpp` — GLFW-owned GL 4.6 shared-context creation;
  GLX on Linux X11/XWayland, WGL on Windows
- `GaussianSplatBuffers.h/.cpp` — shared SSBO/UBO size planning for backends
- `GaussianSplatDataPacking.h/.cpp` — CPU → GPU data packing (std140/std430)
- `FilamentResourceManager.h/.cpp` — `CreateImportedTexture()` / `CreateImportedMTLTexture()`
  for zero-copy import
- `FilamentView.h/.cpp` — `EnableViewCaching()` invalidation fix before freeing color_buffer_
- `FilamentScene.h/.cpp` — `InvalidateGaussianSplatOutput()` forwarding
- `FilamentRenderer.h/.cpp` — frame schedule and GS output forwarding
- `FilamentEngine.cpp` — pre-Filament shared context setup

### Shader files (9 SPIR-V programs)

| Index | File | Pass |
|---|---|---|
| 0 | `gaussian_project.comp` | Projection, tile rect encoding |
| 1 | `gaussian_prefix_sum.comp` | Tile prefix sum, counter write |
| 2 | `gaussian_scatter.comp` | Tile-entry scatter |
| 3 | `gaussian_composite.comp` | Depth-aware compositing |
| 4 | `gaussian_radix_sort_keygen.comp` | Keygen: `(tile_id<<D)|((depth<<1)>>T)`, T from `limits.w` |
| 5 | `gaussian_radix_sort_histograms.comp` | Radix histogram |
| 6 | `gaussian_radix_sort.comp` | Radix scatter |
| 7 | `gaussian_radix_sort_payload.comp` | Payload rearrangement |
| 8 | `gaussian_compute_dispatch_args.comp` | Indirect dispatch counts and `RadixSortParams` (no CPU readback) |

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
| Dynamic sort key split (T/D) | Adapts tile/depth bit allocation to actual tile count; sign-bit stripping recovers one free depth bit; 1080p gets 19 depth bits vs 16 previously |
| Pre-destroy invalidation on resize | Prevents Filament handle use-after-free during maximize/resize |
