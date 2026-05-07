# 3D Gaussian Splatting Rendering in Open3D

## Overview

Open3D supports real-time 3D Gaussian Splatting (3DGS) rendering through a GPU compute
pipeline that runs alongside the Filament-based visualization engine.  The compute pipeline
projects, sorts, and composites Gaussian splats into a color image.  A shared depth texture
lets the composite shader reject splats behind Filament-rendered mesh geometry for correct
per-splat occlusion.  Multiple Gaussian scenes are supported.  The full pipeline also runs in
offscreen `RenderToImage` / `RenderToDepthImage` captures.

**Supported platforms**: Linux X11/GLX (including Wayland via XWayland),
Windows/WGL, macOS/Metal.

---

## Algorithm: 3D Gaussian Splatting

### Gaussian Primitive

Each Gaussian $i$ stores:

* **Position** ($\mu_i$): $3 \times 1$ vector.
* **Rotation** ($q_i$): normalized unit quaternion $[w, x, y, z]$.
* **Scale** ($s_i$): $3 \times 1$ linear-space vector $[s_x, s_y, s_z]$.  PLY files store
  log-scales, exponentiated at load time; SPLAT files store linear scales directly.
* **Opacity** ($\alpha_i$): sigmoid-mapped scalar.  Sigmoid is applied once at CPU packing
  time, not per-frame in the shader.
* **Spherical Harmonics** ($SH_i$): up to degree 3 (48 coefficients per color channel).

3D covariance: $\Sigma = R S S^T R^T$ where $R$ comes from $q_i$ and $S = \text{diag}(s_i)$.

### Projection

1. **View-space mean**: $x_\text{view} = W \mu_i$.
2. **2D covariance**: $\Sigma' = J W \Sigma W^T J^T$ (Jacobian $J$ of perspective projection).
   A $0.3 \times I_{2 \times 2}$ low-pass filter is added to ensure sub-pixel splats cover
   at least one pixel.
3. **Cull**: splats behind the camera or with negligible 2D radius are discarded.

### Tile-Based Sort

Screen is divided into $16 \times 16$ px tiles.  Each splat that overlaps $N$ tiles produces
$N$ sort entries.  Each entry carries a 32-bit sort key:

$$\text{key} = (\text{tile\_index} \ll D) \;\Big|\; ((\text{depth\_key} \ll 1) \gg T)$$

where $T = \lceil \log_2(\text{tile\_count}) \rceil$ (tile bits) and $D = 32 - T$ (depth bits).
Details in **Sort Key Layout**.  A 4-pass LSD radix sort orders entries by tile then depth.

### Rasterization (Composite)

For each tile (one compute workgroup), each pixel $(x, y)$:

1. **Cooperative load**: sort entries fetched from shared memory in batches of 256.
2. **Influence**: $G_i(x,y) = \exp\!\left(-\tfrac{1}{2} d^T (\Sigma'_i)^{-1} d\right)$,
   $d = [\text{pixel} - \text{center}_i]$.
3. **Front-to-back alpha blend**:
   $C \mathrel{+}= C_i \cdot (\alpha_i G_i \cdot T)$, $T \mathrel{\times}= (1 - \alpha_i G_i)$.
4. **Early exit**: stop when $T < 1/255$.

Scene-depth occlusion: splats with $\text{linear\_depth} \ge \text{scene\_linear}_{01}$
(behind mesh geometry) are skipped.  The composite shader writes per-splat depth in Filament's
reversed-Z convention for downstream readback compatibility.

---

## Architecture

### Rendering Pipeline

The pipeline splits into two GPU stages: Stage A (projection + sort) and Stage B (composite).

**Non-Apple** (Filament OpenGL + Vulkan compute):

Filament uses an **OpenGL** backend — the only zero-copy texture-sharing path on
Linux/Windows.  GS compute runs on a separate **Vulkan** compute queue.  Stage A is submitted
fire-and-forget (no CPU wait) so Vulkan geometry overlaps Filament rasterization.  Stage B
runs after Filament completes because it needs the scene depth.

```
BeginFrame:
  1. GaussianSplatRenderer::BeginFrame()
  2. engine_.flushAndWait()    -- drain prior Filament (GL) work; shared textures idle
  3. Stage A (geometry)        -- VK Submit + fence signal, NO CPU WAIT (fire-and-forget)
  4. renderer_->beginFrame()

Draw:
  5. Filament scene draw       -- writes depth to shared GL depth texture
  6. engine_.flushAndWait()    -- depth ready for composite
  7. WaitForGeometryPass()     -- VK fence wait (usually no-op; geometry done during step 5)
  8. Stage B (composite)       -- VK Submit + wait; writes GS RGBA16F
  9. ImGui                     -- base + splat overlay
```

The two mandatory CPU stalls (`flushAndWait`) cannot be eliminated without modifying Filament.

**Apple** (Metal): GS composite runs after `renderer_->endFrame()` on the same Metal queue
ordering as Filament's submit.  The first `Draw()` shows the previous frame's composite;
`SetOnAppleGaussianCompositeComplete` → `PostRedraw()` schedules a second draw so updated
splats appear without a user input event.

`FilamentRenderToBuffer::Render()` mirrors the same `#if defined(__APPLE__)` ordering.

### Depth-Aware Compositing (Zero-Copy)

The GS compute context and Filament share the same GLX context group.  GL texture handles
are valid in both contexts — no CPU copies.

**Shared depth texture**: a `GL_DEPTH_COMPONENT32F` texture is created in the helper
context, imported into Filament as `DEPTH_ATTACHMENT | SAMPLEABLE`, and set as the view's
depth attachment.  Filament writes depth; the composite shader reads it at binding 14.

**Shared color texture**: a `GL_RGBA16F` texture is imported into Filament as `SAMPLEABLE`.
The composite shader writes to it via `imageStore`.  ImGui blends it over the Filament scene
color buffer (SrcAlpha / 1−SrcAlpha).

**MSAA**: disabled for GS views.  Filament asserts `!msaa.enabled || !hasSampleableDepth()`.
3DGS uses Gaussian kernel anti-aliasing instead.

### Multi-Object Scenes

`FilamentScene` maintains two buffer levels:

- **`per_object_gs_attrs_`** (per-`AddGeometry` call): packed per-object splat data.
- **`merged_gs_attrs_`**: single GPU buffer consumed by the pipeline, rebuilt by
  `RebuildMergedGaussianData()` on every add, remove, or update.  Concatenates all per-object
  data and writes a bit-packed `visibility_mask` (1 bit per splat).

`ShowGeometry` patches the object's mask slice in-place (no full rebuild) and calls
`MarkGeometryChanged()`.  The project shader tests the mask bit before writing sort entries;
hidden splats produce no sort entries.

### Scene-Depth Fast Path

Scene depth is always allocated to keep render-target topology stable.  When no mesh
geometry is visible, the composite shader gates occlusion testing via
`depth_range_and_flags.w` (1.0 = scene depth present, 0.0 = absent), so no
render-target re-setup is needed when mesh visibility toggles.

### Offscreen Rendering

`FilamentRenderToBuffer` mirrors the interactive pipeline:
- `EnableViewCaching(true)` for the offscreen view (valid Filament color buffer for zero-copy setup).
- `RequestRedrawForView` before each `Render()` forces the GS pipeline to re-run even
  when the scene and camera are unchanged.
- **Color readback**: two parallel `readPixels` (Filament RGBA+UBYTE, GS RGBA+FLOAT) then
  a second `flushAndWait()`; CPU `BlendPremultipliedSplatOverRgb8` composites the overlay.
- **Depth readback** (GL): `gaussian_depth_merge.comp` merges GS + Filament depth into a
  normalised R16UI texture; `ReadMergedDepthToUint16Cpu` reads it via `glGetTexImage`.
- **Metal constraint**: `readPixels` always uses RGBA+UBYTE (Metal has no native RGB format);
  alpha is stripped when `n_channels_ == 3`.

### Shared GL Context Strategy

The compute context must be created **before** `Engine::create()`.  GLX/WGL context sharing
is set at creation time and cannot be added retroactively.

1. `FilamentEngine.cpp` calls `GaussianSplatOpenGLContext::GetInstance().InitializeStandalone()`
2. `InitializeStandalone()` creates a hidden GLFW OpenGL 4.6 helper window.
3. The native context handle is passed to `Engine::create()` as `sharedGLContext`.
4. Filament's GL platform creates its own context sharing with that native handle
   (`PlatformGLX` on Linux, `PlatformWGL` on Windows).
5. Both contexts share the same GL object namespace; texture handles are valid in both.

### Backend Abstraction

```
GaussianSplatRenderer::Backend (abstract)
├── GaussianSplatVulkanBackend      — Linux + Windows (Vulkan compute; GL_EXT_memory_object
│                                     for zero-copy with Filament OpenGL)
├── GaussianSplatMetalBackend       — macOS (Metal compute)
└── GaussianSplatPlaceholderBackend — logs once per view, returns false
```

Each backend implements `RenderGeometryStage` (Stage A) and `RenderCompositeStage` (Stage B).
Stage A submits fire-and-forget on non-Apple so Vulkan geometry overlaps Filament's draw.
On Metal, Stage B runs after `endFrame()` so Filament's depth is submitted first.

---

## Sort Key Layout

Each sort key packs `tile_index` and `depth` into a single 32-bit uint with a dynamic bit
split computed from the actual tile count:

```
bits 31..D   tile_index   (T = ceil(log2(tile_count)) bits, clamped to [1,31])
bits D-1..0  depth field  (D = 32-T bits)
```

```
key = (tile_index << D) | ((depth_key << 1u) >> T)
```

`depth_key = floatBitsToUint(norm_depth)` where
`norm_depth = (linear_depth - near) / (far - near)` — normalized linear depth in [0,1].

**Why linear depth**: uniform sort-key resolution across the full depth range.  Inverse depth
gives a $1/d^2$ distribution that crowds all key space near the camera.

**`<< 1` (sign-bit strip)**: `norm_depth` is in [0,1] so the IEEE 754 sign bit is always 0;
stripping it reclaims one free depth bit.

**`floatBitsToUint` logarithmic tilt**: IEEE 754 has more representable values near zero,
so slightly more sort keys are allocated near the camera where sort errors matter most.

$T$ is computed CPU-side from the tile count and stored in `GaussianViewParams.limits.w`.

| Viewport | Tiles (16×16) | T (tile bits) | D (depth bits) |
|----------|--------------|--------------|----------------|
| 1080p    | 8 160        | 13           | 19             |
| 4K       | 32 400       | 15           | 17             |
| 8K       | 129 600      | 17           | 15             |

The 4-pass radix sort operates on all 32 bits (8 bits per pass); the key layout change
requires no sort logic updates.

---

## Shaders

Six GLSL compute shaders are compiled offline to SPIR-V (`-V --target-env vulkan1.3`) by
`open3d_add_compute_shaders` and to Metal Shading Language (MSL) via SPIRV-Cross.  All
compiled artifacts are placed in `resources/gaussian_splat/`.

| Index | File | Purpose |
|-------|------|---------|
| 0 | `gaussian_project.comp` | Projects splats to 2D, writes tile sort entries via per-subgroup atomic |
| 1 | `gaussian_composite.comp` | Depth-aware rasterization; binary-search per tile; outputs RGBA16F + depth |
| 2 | `gaussian_radix_sort_histograms.comp` | Builds per-digit histograms for one radix pass |
| 3 | `gaussian_radix_sort_scatter.comp` | Scatters key-value pairs using subgroup prefix-sum |
| 4 | `gaussian_compute_dispatch_args.comp` | Writes all indirect dispatch counts and `RadixSortParams` GPU-side (no CPU readback) |
| 5 | `gaussian_depth_merge.comp` | Merges GS + Filament depth → normalised R16UI for offscreen readback |

`gaussian_radix_sort_scatter.comp` uses `GL_KHR_shader_subgroup_{basic,arithmetic,ballot,shuffle}`
(Vulkan 1.3 subgroup arithmetic).  The Apple Metal build fixes the SIMD group size to 32
(Apple Silicon SIMD width) via `--msl-fixed-subgroup-size 32` in SPIRV-Cross so the compiler
can treat `gl_SubgroupSize` as a constant.

---

## Data Packing

### Per-Frame View UBO (`GaussianViewParams`, std140, binding 0)

288-byte struct packed CPU-side by `PackGaussianViewParams` every frame.

| Field | Meaning |
|-------|---------|
| `scene.z` | Antialias flag (0 = off, 1 = density compensation) |
| `limits.x` | Tile-entry capacity allocated for the frame |
| `limits.y` | `RenderConfig::max_tiles_per_splat` |
| `limits.z` | `RenderConfig::max_tile_entries_total` |
| `limits.w` | $T$ (tile bit count for sort key split) |
| `depth_range_and_flags.w` | 1.0 = scene depth present, 0.0 = absent |

### Per-Splat GPU Buffers

Packed once per scene change by `PackGaussianSplatAttrsDirect`:

| Buffer | Binding | GPU type | B/splat | CPU encoding |
|--------|---------|----------|---------|--------------|
| `positions` | 1 | `vec4` fp32 | 16 | direct copy |
| `scales` (linear) | 2 | `uvec2` fp16×4 | 8 | `PackHalf2` ×2 |
| `rotations` | 3 | `uint` snorm8-biased×4 | 4 | `PackSnorm8x4(w,x,y,z)` |
| `dc_opacity` | 4 | `uvec2` fp16×4 | 8 | `PackHalf2` ×2; sigmoid pre-applied |
| `sh_coefficients` | 5 | `uvec2` fp16 | 0/24/48 | `PackHalf2` pairs; degree-dependent |
| `visibility_mask` | 15 | `uint32[]` bitfield | ~0.125 B/splat | `ceil(N/32)` uint32 words |
| **Total** | | | **36–84 B/splat** | |

### Per-Splat Intermediate Buffers

| Buffer | Binding | Layout | Purpose |
|--------|---------|--------|---------|
| `ProjectedComposite` | 6 | 32 B/splat: fp16 center + fp32 depth + fp32 alpha + rgba8 + `vec4 inv_basis` | Written by project; read by composite |
| `sort_keys` | 7 | `uint32` | LSD radix sort keys (ping-pong) |
| `sort_values` | 8/9 | `uint32` splat index | LSD radix sort values (ping-pong) |
| `histogram` | 10 | `uint32[WG × 256]` | Per-workgroup digit histograms |
| `dispatch_args` | 11 | `uint32[8×3]` | Indirect dispatch args (4×histogram + 4×scatter) |
| `RadixSortParams` | 14 | std140, 16 B ×4 passes | Per-pass digit shift and element count |

`counters_buf` (binding 10, `GaussianGpuCounters`): GPU→CPU diagnostic channel — total entries,
error flags, tile count, splat count.

### Anti-aliasing / Density Compensation

The projection pass always adds $+0.3 \cdot I_{2\times2}$ to the projected covariance
(low-pass regulariser).  With `RenderConfig::antialias = true`, the projection shader cancels
the artificial brightness increase with:

$$\text{compensation} = \sqrt{\frac{\det(\Sigma_{\text{orig}})}{\det(\Sigma_{\text{blurred}})}}$$

`alpha *= compensation`.  The ratio is clamped before the square root to handle degenerate
(zero-area) splats.

### Runtime Capacity Limits and Error Flags

`RenderConfig` exposes two runtime safety knobs:

- `max_tiles_per_splat` — per-splat budget for estimating sort buffer size.
- `max_tile_entries_total` — hard ceiling for sort keys, values, and tile entries.

Both are forwarded to shaders via `GaussianViewParams.limits`.

GPU error bits in `counters_buf.error_flags`:

- Bit 0: tile-entry overflow in the scatter pass; excess entries dropped.
- Bit 1: dispatch / sort count clamped in `gaussian_compute_dispatch_args.comp`.

The pass runner downloads this bitmask once after GPU work and logs each warning once per view.

---

## Geometric Transforms for Gaussian Splat PointClouds

`t::geometry::PointCloud::Rotate`, `Scale`, and `Translate` correctly update all Gaussian
attributes when `IsGaussianSplat()` is true.  `Transform(4×4)` warns for GS clouds because
a general matrix may be non-orthogonal.

| Operation | Positions | `rot` | linear `scale` | `f_dc` | `f_rest` |
|-----------|-----------|-------|----------------|--------|----------|
| `Translate` | Updated | — | — | — | — |
| `Rotate(R, c)` | Updated | Composed with $q_R$ | — | — | IR-rotated |
| `Scale(s, c)` | Updated | — | Multiplied by $|s|$ | — | Odd-degree blocks negated if $s<0$ |
| `Transform(4×4)` | Updated | **unchanged** (warning) | **unchanged** (warning) | **unchanged** | **unchanged** |

### Covariance and Quaternion Semantics

Covariance $\Sigma = R_q S^2 R_q^T$.  After `Rotate(R)`: $\Sigma' = R \Sigma R^T$, so
$q' = q_R \cdot q_\text{old}$ (quaternion left-multiply; no eigendecomposition needed).

After `Scale(s)`: $\Sigma' = |s|^2 \Sigma$.  Negative uniform scale is the improper transform
$-I$ (point inversion) followed by positive scaling.  The quaternion is unchanged; linear
scales multiply by $|s|$; SH picks up the parity of point inversion.

### SH Rotation — Ivanic–Ruedenberg (IR) Algorithm

`f_rest` layout: `{N, Nc, 3}` where `Nc = (sh_degree+1)^2 − 1`, last axis = RGB.
Coefficient ordering: $k = l^2 + l + m - 1$ with $l = 1 \ldots \text{sh\_degree}$, $m = -l \ldots l$.

The shader evaluates degree-1 SH as `coeffs[0]*dir.y + coeffs[1]*dir.z + coeffs[2]*dir.x`
(ordering: $m=-1, 0, +1$ → Cartesian $y, z, x$).  The IR degree-1 rotation matrix uses
permutation `idx = {1, 2, 0}`:

$$R_1[i,j] = R_{\text{SO3}}[\text{idx}[i],\, \text{idx}[j]]$$

Higher degrees derived recursively using Ivanic–Ruedenberg $u, v, w$ weights.  The same
$R_l$ matrix is applied to all three RGB channels independently.

For point inversion (`Scale(s < 0)`): $Y_{lm}(-d) = (-1)^l Y_{lm}(d)$.  `f_dc` (degree 0)
is unchanged; odd-degree blocks in `f_rest` ($l = 1, 3, \ldots$) are negated in place.

Degrees 1 and 2 are evaluated at render time (`EvaluateShDegree1/2` in
`gaussian_project.comp`).  Degree 3 coefficients are rotated in the stored tensors for
consistency even though the shader does not evaluate them.

**Reference**: Ivanic & Ruedenberg (1996) + 1998 erratum.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Filament OpenGL backend on Linux/Windows | Only zero-copy texture-sharing path (via GL context group sharing). |
| Standalone GL context before `Engine::create()` | GLX/WGL sharing is set at context creation time; cannot be added retroactively. Once Filament's driver thread owns its context, sharing is impossible. |
| Force GLX on Linux (including Wayland sessions) | Filament v1.54.0 uses `PlatformGLX` unconditionally on Linux. An EGL context passed as `sharedGLContext` causes `glXQueryContext` to fail. `GLFW_PLATFORM_X11` is forced; XWayland provides GS functionality on all Wayland compositors. |
| Vulkan compute instead of GL compute | GL compute shaders have limited/no subgroup support on Intel hardware. Vulkan compute provides full `VK_KHR_shader_subgroup` on all major vendors (NVIDIA, AMD, Intel), enabling subgroup-optimized sort and projection shaders. |
| Fire-and-forget geometry stage | `EndGeometryPass()` submits the Vulkan command buffer and signals a fence without waiting.  Vulkan geometry overlaps Filament's `beginFrame()` and scene draw.  `WaitForGeometryPass()` before composite is typically a no-op because geometry finishes during Filament's draw. |
| `VK_QUEUE_FAMILY_EXTERNAL` for GL–Vulkan handoff | Images shared with OpenGL via `EXT_external_memory` require queue-family ownership acquire/release in every composite command buffer (`VK_QUEUE_FAMILY_EXTERNAL` ↔ compute queue). `VK_QUEUE_FAMILY_IGNORED` is only valid without external APIs; using it for shared images causes `VK_ERROR_DEVICE_LOST` on strict drivers (Windows AMD/Intel). `engine_.flushAndWait()` provides CPU-side ordering but does not substitute for this GPU-side ownership transfer. |
| `engine_.flushAndWait()` for synchronization with Filament | GL semaphore objects (`GL_EXT_semaphore`) are shared across contexts, but signal/wait commands must be issued on a specific context's command stream. We cannot inject these into Filament's driver thread without modifying Filament internals. CPU fence waits are the only viable synchronization mechanism. |
| Acquire barrier uses `oldLayout = UNDEFINED` | Standard external-acquire pattern (Vulkan spec §12.7.4). We do not track the image's previous layout because Filament (OpenGL) manages it; treating it as undefined is always valid. |
| Release barrier transitions to `GENERAL` | Releases ownership back to the external GL consumer without imposing a Vulkan layout constraint. |
| Prefer graphics+compute queue family | The same hardware engine (Intel RCS, AMD GFX) as the OpenGL context. Dedicated compute-only queues (Intel CCS) can behave differently for the same SPIR-V and have caused hangs on some driver/hardware combos. |
| Two-stage compute split (Stage A / Stage B) | Stage A (project + sort) can overlap Filament rasterization on the GPU. Stage B (composite) must wait for Filament's depth output. Splitting the submit is the minimum required for GPU-level overlap without modifying Filament. |
| Shared sampleable depth texture | Zero-copy: Filament writes depth, composite reads it without any CPU staging. The MSAA restriction is acceptable because 3DGS uses Gaussian kernel anti-aliasing. |
| Scene depth always allocated | Avoids render-target topology changes when mesh visibility toggles; the composite shader gates occlusion via `depth_range_and_flags.w` at runtime instead. |
| Normalized linear depth for sort keys | Uniform $\Delta d$ per sort-key interval across the full depth range. Inverse depth gives a $1/d^2$ distribution crowding all key space near the camera. `floatBitsToUint` provides a free log-density tilt toward the near field. |
| Dynamic T/D sort-key split | Adapts tile/depth bit allocation to the actual tile count. The sign-bit strip (always zero for `norm_depth` ∈ [0,1]) reclaims one free depth bit without cost. |
| Binding 14 reuse (radix UBO + scene depth sampler) | Safe because the radix UBO (passes 3–10) and the scene depth sampler (pass 11 composite) are never active in the same dispatch. |
| GPU-side dispatch args | `gaussian_compute_dispatch_args.comp` writes all indirect dispatch counts and `RadixSortParams` GPU-side after projection, eliminating a CPU readback stall. |
| Subgroup-batched atomic in project | `WriteSortEntries()` uses `subgroupAdd` / `subgroupExclusiveAdd` to batch the global counter increment: one `atomicAdd` per subgroup (~32 lanes) instead of per tile-entry, reducing global atomic traffic by ~32×. |
| Work-stealing composite threads | Each composite workgroup atomically claims tiles from a global counter until all tiles are processed.  Binary search on `sort_keys` finds each tile's entry range inline, eliminating a separate tile-boundary pass and its intermediate buffer. |
| Packed `ProjectedComposite` (32 B/splat) | One 32 B SSBO read per splat in composite instead of two separate reads from former split bindings.  Halves L2 cache lookups for the composite pass. |
| Compressed input SSBOs | `scales` as fp16×4 (8 B), `rotations` as snorm8-biased uint (4 B), `dc_opacity` as fp16×4 (8 B), SH as fp16 (degree-dependent).  Total 36–84 B/splat vs. the previous 160 B. |
| Sigmoid applied CPU-side | Eliminates a per-splat per-frame transcendental `exp()` in the projection shader; computed once at packing time. |
| Bit-packed visibility mask | 1 bit per splat (0.125 B/splat).  The project shader reads a single `uint32` word per 32 splats; masked splats write no sort entries. |
| Pre-destroy invalidation on resize | `InvalidateGaussianSplatOutput()` tears down the GS render target before `FilamentView` frees `color_buffer_`, preventing use-after-free during maximize/resize. |
| Metal `SetOnAppleGaussianCompositeComplete` + `PostRedraw` | Composite runs after `endFrame()`; without a `PostRedraw()`, the first frame shows no splats until the next user event.  The callback schedules a deferred redraw. |

---

## File Inventory

### Core implementation (`cpp/open3d/visualization/rendering/gaussian_splat/`)

| File | Purpose |
|------|---------|
| `GaussianSplatRenderer.h/.cpp` | Backend interface; per-view output lifecycle; `BeginFrame()`; `RenderCompositeStage()`; `ReadMergedDepthToUint16Cpu()` |
| `GaussianSplatDataPacking.h/.cpp` | CPU→GPU data packing (std140/std430); `GaussianGpuBufferSizes`; `PackGaussianViewParams`; `PackGaussianSplatAttrsDirect` |
| `ComputeGPU.h` | `ComputeProgramId` enum; `GaussianSplatGpuContext` abstract base; `GpuComputeFrame` / `GpuComputePass` RAII helpers; `kGsShaderNames[]` |
| `ComputeGPUVulkan.h/.cpp` | Vulkan `GaussianSplatGpuContext`: pipeline management, SSBO/UBO binding, command buffer lifecycle, fence-based geometry sync |
| `GaussianSplatVulkanInteropContext.h/.cpp` | Headless Vulkan instance + device; allocates exportable `VkImage` memory, imports into GL via `GL_EXT_memory_object` |
| `GaussianSplatVulkanBackend.h/.cpp` | Vulkan `GaussianSplatRenderer::Backend` for Linux/Windows |
| `ComputeGPUMetal.mm` | Metal `GaussianSplatGpuContext`: buffer management, pipeline dispatch, barriers, texture ops |
| `GaussianSplatMetalBackend.mm` | Metal `GaussianSplatRenderer::Backend`; acquires Filament `MTLDevice`/queue; creates and imports `MTLTexture` targets |
| `GaussianSplatPassRunner.h/.cpp` | Backend-agnostic geometry + composite pass sequence (shared by Vulkan and Metal); dispatch group sizes computed inline |
| `GaussianSplatOpenGLContext.h/.cpp` | GLFW-owned GL 4.6 shared-context creation (GLX on Linux, WGL on Windows) |
| `shaders/` | Compute shader sources (`.comp`) |

### Filament integration (`cpp/open3d/visualization/rendering/filament/`)

| File | Purpose |
|------|---------|
| `FilamentNativeInterop.h/.mm` | Retrieves Filament `MTLDevice` and `MTLCommandQueue` from `PlatformMetal` |
| `FilamentResourceManager.h/.cpp` | `CreateImportedTexture()` / `CreateImportedMTLTexture()` for zero-copy import |
| `FilamentView.h/.cpp` | `EnableViewCaching()` invalidation fix; `GetRenderTargetHandle()` for offscreen readback |
| `FilamentScene.h/.cpp` | `per_object_gs_attrs_` / `merged_gs_attrs_`; `RebuildMergedGaussianData()`; `HasNonGaussianVisibleGeometry()` |
| `FilamentRenderToBuffer.h/.cpp` | GS pipeline mirror; parallel `readPixels`; `BlendPremultipliedSplatOverRgb8` CPU blend |
| `FilamentRenderer.h/.cpp` | Frame schedule and GS output forwarding; Apple `SetOnAppleGaussianCompositeComplete` |
| `FilamentEngine.cpp` | Pre-Filament shared context setup |
| `Window.cpp` | Registers composite-complete callback → `PostRedraw()` (Metal first-frame fix) |

### Shader files (`shaders/`)

| File | `ComputeProgramId` | Pass |
|------|--------------------|------|
| `gaussian_project.comp` | `kGsProject` | Projection, sort-entry emission |
| `gaussian_composite.comp` | `kGsComposite` | Depth-aware rasterization |
| `gaussian_radix_sort_histograms.comp` | `kGsRadixHistograms` | Per-digit histogram |
| `gaussian_radix_sort_scatter.comp` | `kGsRadixScatter` | Key-value scatter (subgroup prefix-sum) |
| `gaussian_compute_dispatch_args.comp` | `kGsDispatchArgs` | GPU-side indirect dispatch args |
| `gaussian_depth_merge.comp` | `kGsDepthMerge` | GS + Filament depth → R16UI |

### Tests and examples

| File | Purpose |
|------|---------|
| `cpp/tests/visualization/rendering/GaussianSplatRender.cpp` | `RenderToImage` golden PNG test (36×20, `AllClose atol=5`); `OPEN3D_TEST_GENERATE_REFERENCE=1` regenerates reference |
| `examples/cpp/GaussianSplat.cpp` | Interactive viewer with red sphere for depth compositing testing |
