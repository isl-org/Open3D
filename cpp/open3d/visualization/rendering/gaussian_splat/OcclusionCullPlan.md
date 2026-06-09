# Plan: Prior-Frame Depth Reprojection Occlusion Culling for 3DGS

Status: PLAN (not yet implemented). Owner: TBD.

## Goal (LOCKED)

Increase 3DGS rendering throughput by culling, in the projection pass, splats that
are fully occluded by geometry that was visible in the previous frame. Culling must
be **conservative**: a splat that is (even partially) visible in the current frame
must never be culled, so the rendered image is unchanged within tolerance.

## Requirements (LOCKED)

1. Save the merged (GS + Filament scene) front-surface depth of frame *N* to a
   persistent per-view GPU resource, together with frame *N*'s camera transform and
   near/far.
2. At the start of frame *N+1* (Stage A, before the project pass), forward-reproject
   the saved frame-*N* depth into frame *N+1* screen space and build a **max-depth
   pyramid** (each coarser level = max of the 2×2 finer texels) — a Hi-Z buffer.
3. In `gaussian_project.comp`, use the Hi-Z pyramid to cull a splat when its entire
   screen footprint lies strictly behind the pyramid (i.e. behind every reprojected
   occluder in the footprint).
4. Conservative by construction: reprojection holes/cracks and disoccluded regions
   read as "far" so they never cause culling.
5. Feature is opt-in via `RenderConfig` and degrades to a no-op on frame 0 (no prior
   depth) and when disabled. Existing public APIs unchanged.
6. Works on the Vulkan backend (Linux/Windows) and Metal backend (macOS). Pyramid is
   stored in an SSBO (uint) to keep atomics and addressing portable across both.

## Test Method (LOCKED)

- **Correctness (image invariance):** Extend
  `cpp/tests/visualization/rendering/GaussianSplatRender.cpp`. Render the existing
  golden scene with occlusion culling ON and OFF across several camera poses
  (including the first frame and after a large camera jump) and assert the two color
  images match within the current golden tolerance (`AllClose atol=5`). Regenerate
  golden with `OPEN3D_TEST_GENERATE_REFERENCE=1` only if intentionally changed.
- **Effectiveness:** Add a scene with a large opaque occluder (e.g. reuse the red
  sphere in `examples/cpp/GaussianSplat.cpp`) in front of a dense splat cloud. Read
  back `gs_counters.total_entries` / `splat_count` via the existing counters channel
  and assert the entry count drops materially (e.g. ≥ X%) with culling ON vs OFF on
  a steady camera, while the image still matches.
- **Safety on motion:** Assert no visible artifacts after a large single-frame camera
  delta (disocclusion case) — covered by the multi-pose image-invariance test above.
- Run the smallest relevant C++ test target: `./bin/tests --gtest_filter=*GaussianSplat*`.

---

## Design Refinements (answers to review questions)

### R1. Occluder threshold must differ from the visible-depth threshold (correctness)

`gaussian_composite.comp` writes its visible depth the moment
`transmittance < kSurfaceThreshold = 0.5`. That "50%-surface" is **not** a safe
occlusion boundary: at T=0.5 half the light still passes, so splats behind it are
still visible. Culling against it would change the image.

The occluder/prior depth must be captured at a **stricter** transmittance — the depth
at which `T` first drops below `kOccluderThreshold` (everything behind then
contributes < that fraction). Pick it to match the golden tolerance (`atol=5/255 ≈
0.02`), e.g. `kOccluderThreshold = 1/255` (or the existing `kMinTransmittance =
1/4096` to be maximally safe). This is a **separate, cheap second capture inside the
existing front-to-back loop** — one extra compare + store, no extra pass:

```glsl
// existing visible-depth capture (unchanged):
if (!depth_written && transmittance < kSurfaceThreshold) { out_norm_depth = s_depth[i]; depth_written = true; }
// NEW occluder-depth capture (stricter):
if (!occluder_written && transmittance < kOccluderThreshold) { occluder_norm_depth = s_depth[i]; occluder_written = true; }
```

Pixels that never reach `kOccluderThreshold` keep `occluder_norm_depth = far` → never
an occluder (safe). So `gaussian_composite.comp` gains a second R32F output
(`out_occluder_depth`) alongside the existing visible depth.

### R2. Reuse `gaussian_depth_merge.comp` for the "save" step (no new save shader)

Extend the existing merge shader to take the new occluder depth (GS) and the Filament
scene depth and write a **persistent linear** `prior_depth_tex` (the occluder surface
of frame N). Mesh geometry is fully opaque, so merging the scene depth in is correct
and free. Decouple this from `wants_depth_readback` so it runs whenever the feature is
on. When `has_scene_depth == false`, the composite occluder depth IS the saved depth
and the merge can be skipped entirely (a straight copy/alias).

### R3. Can DepthReproject and HiZ-Reduce be folded into the merged shader? No — but the set stays minimal

Three operations live at two different pipeline points / cameras:

| Op | When | Camera | Access pattern |
|----|------|--------|----------------|
| save (merge) | end of frame N | N | per-pixel gather |
| reproject | start of frame N+1 | N → N+1 | **scatter** (atomicMin) |
| HiZ reduce | start of frame N+1 | N+1 | gather (max 2×2) |

- **merge ≠ reproject/reduce**: merge happens at end of frame N and does not know
  frame N+1's camera, so it cannot be the same dispatch.
- **reproject ≠ reduce in one dispatch**: reproject is a *scatter* that must fully
  populate pyramid level 0 before any 2×2 *max* reduce can read neighbours →
  requires a global barrier between them → cannot be one dispatch. Folding the max
  into the scatter (scatter straight into coarse levels with `atomicMin`) is
  **unsafe**: per coarse texel that yields the *nearest* scattered occluder, but the
  conservative test needs the *farthest* (max) over the region — using min would
  over-cull and drop visible splats.

**Can reproject + reduce share one shader file (branch on a `mode` argument)?** No
benefit. They still need two dispatches (the global barrier above is unavoidable), and
their bindings/local-sizes/logic are disjoint — scatter binds a saved-depth sampler +
the uint SSBO (out), reduce binds the uint SSBO (in) + mip storage images (out). A
merged shader would declare the union of bindings and leave half unused per dispatch,
breaking the one-`ComputeProgramId`-per-fixed-binding-table convention. Keep them as
two separate shaders.

Net shader budget: **2 changed** (`gaussian_composite.comp`, `gaussian_depth_merge.comp`)
+ **2 new** (`gaussian_depth_reproject.comp`, `gaussian_hiz_reduce.comp`). The reduce
shader builds the **entire pyramid in a single dispatch** (local-only SPD; see R6).

### R4. Vulkan/Metal feature check → uint-SSBO atomics + half-float mipmapped pyramid

The scatter needs an atomic min. Portability constraints:
- **Float image/texture atomics** (`VK_EXT_shader_atomic_float[2]`, `imageAtomicMin`
  on float) are optional on Vulkan and **unsupported on Metal**.
- **Texture atomics in general** are not portable on Metal (atomics are a buffer
  feature there). The existing pipeline already does all atomics in **SSBOs**.

Decision:
- **Scatter target = `uint` SSBO**, full-res, level 0 only. `norm_depth ∈ [0,1] ≥ 0`,
  so `floatBitsToUint` is monotonic and `atomicMin` on the bit pattern == min of the
  floats. Clear to `floatBitsToUint(1.0)` (far). Portable on both backends.
- **Pyramid = half-float (R16F) mipmapped texture** (the user's preference), levels
  0..L. Built by `gaussian_hiz_reduce.comp` in a **single dispatch** (local-only SPD;
  see R6): each workgroup owns one 32×32 source tile, converts the uint SSBO → mip 0
  (`uintBitsToFloat`), then max-reduces the tile through all in-tile levels
  (32→16→8→4→2→1) in shared memory + `subgroupMax`. The project pass reads it with a
  single `textureLod` at the footprint-matched level.
- Mipmapped storage: bind one mip level per storage `imageView` for writes (Vulkan)
  / per-level texture view (Metal); sample the full chain via one sampler. Both
  backends support this without extensions.

**Is half (R16F) sufficient?** Yes, with the epsilon guard. `norm_depth` is
normalized linear [0,1]; R16F resolution near the far plane is ~5e-4, finer near the
camera. The `kOcclusionEpsilon` guard (~1e-3 of range) absorbs it, and the max-reduce
already biases toward *far* (the safe direction). Level 0 keeps full precision in the
uint SSBO, so the per-pixel `min` is exact; lossy half only appears in coarse levels
where conservatism already dominates. **Fallback:** if macOS testing shows artifacts,
switch the pyramid to R32F (or all-uint) — addressing is unchanged.

### R5. Further simplifications adopted
- Occluder depth piggybacks on the composite loop (no separate occluder pass).
- Save step reuses `gaussian_depth_merge.comp` (no new save shader); skipped when no
  mesh depth is present (copy/alias instead).
- Cap the pyramid at the level covering the largest plausible footprint (e.g. 32 px =
  the in-tile top level) instead of reducing to 1×1 — see R6.
- Project-pass read is a single conservative tap (defer 4-tap tightening).
- Reuse the existing `composite_depth_tex` allocation/resize pattern for the new
  persistent textures and the uint scatter SSBO.

### R6. Single-dispatch pyramid build (local-only Single-Pass Downsampler)

The pyramid is built in **one dispatch**, no per-level loop, no atomics, no
globally-coherent memory:

- **One 32×32 source tile per workgroup.** (The GS *rasterization* tile is 16×16 —
  design doc — but the SPD reduction tile is independent; 32×32 covers 6 levels
  32→16→8→4→2→1 and matches the 32-lane subgroup width used elsewhere, incl. Apple's
  `--msl-fixed-subgroup-size 32`.)
- Each workgroup: load its 32×32 tile (convert uint SSBO → mip 0 with
  `uintBitsToFloat`), then max-reduce **in shared memory** level by level with a
  `barrier()` between LDS levels (and `subgroupMax` within a subgroup), writing mips
  0..5 of *its own tile*. Tiles are independent at these levels (no cross-tile reads),
  so **no global barrier and no atomic counter are needed** (unlike full SPD).
- **Cap at the in-tile top level (mip 5 = 32 px footprint).** The project pass never
  samples a coarser level than this (footprint cap, R5), so the cross-tile tail of
  classic SPD — the only part that needs the global atomic + coherent mip-5 image — is
  **omitted entirely**. This is strictly simpler than both full SPD and the per-level
  dispatch loop.
- If a deeper pyramid is ever required (footprint cap removed), upgrade to full SPD
  with an atomic-elected last workgroup reducing mips ≥6. Not needed now.

Workgroup size: 256 threads (16×16), each thread loading a 2×2 quad of the 32×32 tile
(keeps the group ≤ the common 1024-thread limit with headroom and matches a 32-wide
subgroup). Per-level mip writes use one storage image view per level.

### R7. Reprojection cracks vs. real scene gaps (correctness-critical)

**Problem.** Forward scatter writes each source pixel to one integer destination. Under
fractional pixel shifts or magnification, neighbouring destinations skip integer
texels, leaving 1-pixel **cracks** at the cleared value `far`. Because the project
pass reads a **coarse** pyramid level (max over a footprint, up to 32 px), a *single*
crack texel forces that region's max to `far` and **disables culling for the whole
tile** — silently erasing the feature's benefit under smooth zoom/pan.

**The trap.** A crack and a *real* thin scene gap (grass blades, comb teeth,
chain-link — background genuinely visible between foreground elements) are
**indistinguishable from depth alone**: both are a `far` texel flanked by `near`
texels. Any mitigation that fills holes based only on neighbour *presence* will fill
real gaps too → write a near depth into the gap → cull a visible background splat →
**wrong image**. Correctness rule: **a Hi-Z texel used to cull must hold the depth of a
real surface covering that pixel; genuine gaps must read `far`.** Cracks may therefore
cost only *performance* (we keep splats we could have culled), never correctness.

**Rejected mitigations (all corrupt the grass/comb case):**
- **Blind 2×2 dilation** (splat each source pixel into 4 neighbours): spills the
  foreground ~1 px past the silhouette into the gap → culls background seen through it.
- **Valid-children fill in the reduce** (max over non-`far` children only): fills real
  gaps with the surrounding near surface → culls visible background.
- **Half-resolution scatter target:** merges adjacent near (foreground) + far (gap)
  full-res pixels into one texel; the mandatory cross-source `atomicMin` keeps *near* →
  the gap is lost → culls background. (Anti-conservative; same failure as the old M4.)
  *Coarser-but-correct alternative if ever wanted: read pyramid **mip 1** in the
  project pass — free, conservative, but gives no scatter savings and looser silhouette
  culling. Not adopted.*
- **Depth-continuity fill** (fill only where `|d_a−d_b| < τ·d`): correct, but needs a
  scene-dependent threshold `τ`. Superseded by the parameter-free method below.

**Adopted: 2×2 quad projection with MAX-fill (parameter-free, correct by
construction).** Process the previous depth in **2×2 source quads**. Reproject all four
corners; compute the screen-space bounding box of the reprojected quad:
- If the reprojected bbox fits within **3×3** destination texels → the quad did not
  stretch (no disocclusion opened inside it) → **fill its interior**, writing
  `max(d0,d1,d2,d3)` (the **farthest** of the 4 source depths) via `atomicMin` to each
  covered texel.
- Else (stretched > 3×3 → parallax/disocclusion) → **reject**: scatter only the 4
  corners, leave the interior `far` (a crack — safe perf loss).

Why this is correct without any depth threshold:
- The reprojected quad's screen **stretch is the parallax signal**; "compact" ⟺ "no
  disocclusion opened inside the quad" — pure screen geometry, scene-independent.
- **MAX-fill is the keystone.** For a real thin gap captured inside a compact quad
  (e.g. low-parallax rotation), `max` of the corners = the **background** depth → the
  interior is filled with `far` → background splats there satisfy `depth > far+ε` =
  false → **not culled**. The grass/comb case is preserved *by construction*. Filling
  with `min` or interpolated depth would reintroduce the silhouette error.
- For a continuous crack, the 4 corners are ≈ equal → fill ≈ surface depth → splats
  behind it are culled (perf recovered).
- Cross-source merge stays `atomicMin` (nearest occluder wins); only the *intra-quad*
  fill uses max. Residual risk (geometry farther than `max_quad_depth` between corners)
  is ≤1–2 texels and further absorbed by the project-side footprint max-read.

`N = 3` (the 3×3 compactness cap) is **fixed** for simplicity (≈1.5× magnification
coverage; beyond that, cracks are allowed as safe perf loss). It is a pure perf knob —
correctness does not depend on its value once max-fill is used. Granularity bonus: 2×2
quads = ¼ the invocations of per-pixel scatter.

**Complement — `round` not `floor`** for the single-corner destination: relocates the
sample to the nearest texel (never bridges a gap), modestly fewer sub-pixel cracks,
free and safe.

### R8. Scatter atomic-overhead reduction (correct, no resolution change)

Two always-safe, always-conservative reducers — no barriers, no second code path:
- **Background skip:** `if (saved_depth >= far_eps) return;` before the atomic. Far
  source pixels carry no occluder and would never win a `min`; skipping them removes a
  large fraction of atomics outright.
- **Non-atomic pre-check:** `if (encoded < dst[p]) atomicMin(dst[p], encoded);` The
  relaxed load may be stale, but a stale *skip* only ever drops a write that would not
  have lowered the min after a concurrent nearer write → still conservative. Cuts
  redundant atomics exactly in the contended many-to-one case.

**SLM-tiled scatter (LDS pre-reduction then flush) — profiling-gated, not first.**
Reduces global atomics only in heavy **minification** regions; near-zero benefit in
the common ~1:1 smooth-motion case, adds 2 barriers + a divergent second path, and the
flush **still needs global `atomicMin`** (reprojected dest tiles from different source
workgroups overlap). Implement only if profiling shows the reproject pass is
atomic-bound *and* minification-dominated. The background-skip + pre-check above
capture most of the same contention relief at zero structural complexity.

### R9. SLM local-reduce for the scatter — see R8 (deferred, profiling-gated).

---

## Algorithm and Design

### Depth-convention contract (correctness-critical)

Two conventions exist in the current code:

| Quantity | Convention | Source |
|----------|-----------|--------|
| `gaussian_project.comp` `norm_depth` | **linear**, near=0, far=1, larger = farther | computed in project pass |
| `composite_depth_tex` (R32F) / Filament `scene_depth` | **inverse reversed-Z**, near=1, far=0, larger = nearer | `gaussian_composite.comp` `Linear01ToInverse()`, Filament |

The Hi-Z pyramid is defined in the **same space the project pass compares against**:
**linear, near=0, far=1, larger = farther**. The reproject pass converts the stored
inverse depth back to linear when it writes the pyramid. Document this inline in both
new shaders; a convention mismatch is the most likely source of wrong culling.

### Conservative test (why max-depth pyramid)

A splat covers a pixel footprint *F*. It is occluded only if it is behind the visible
surface at **every** pixel in *F*, i.e. `splat_depth_lin > stored_depth_lin` for all
pixels in *F*, which is equivalent to:

```
splat_depth_lin > max_{p in F} stored_depth_lin(p)
```

Hence the pyramid stores the **max linear depth** over each region. Holes read as
`far = 1.0`, so any footprint that touches a hole has `max = 1.0` and is never culled.

Per-pixel vs. per-region combine:
- **Within one pixel** (forward scatter, many prev pixels → one current pixel): keep
  the **nearest** reprojected surface = `min` linear depth (that is the true visible
  occluder). Encoded as uint, use `atomicMin`.
- **Across a region** (pyramid build, 2×2 → 1): keep the **farthest** = `max`
  (conservative). Plain `max` reduction, no atomics.

### Forward reprojection (Step 2)

Process the saved frame-*N* depth in **2×2 source quads** (one invocation per quad;
¼ the invocations of per-pixel scatter). For each quad:
1. Read the 4 saved linear depths; **skip a corner if `d >= far_eps`** (background,
   no occluder — R8 background-skip).
2. Unproject each valid corner to world (`inverse(clip_from_world_N)`), project to the
   current frame (`clip_from_world_{N+1}`), discard corners with `w<=0`; compute each
   current pixel `p'` and current linear depth `d'_lin` from frame-*N+1* near/far.
3. Compute the screen-space bbox of the reprojected corners.
   - **bbox ≤ 3×3 texels (N=3, R7):** fill the interior — for each covered texel,
     `EncodeUint(max(d'_0..d'_3))` written via the pre-checked `atomicMin` (R8). MAX
     of the quad depths is mandatory (preserves real thin gaps; see R7).
   - **bbox > 3×3 (stretched / disocclusion):** scatter only the 4 corners
     (`round` to nearest texel, R7), leave the interior `far` — a safe crack.
4. Cross-source resolution is `atomicMin` (nearest occluder wins); only the intra-quad
   interior fill uses max.

Holes (unwritten texels) and rejected stretched interiors remain at the cleared value
`far` → never cull (safe). Convention: depths are normalized **linear** in [0,1], so
`floatBitsToUint` is monotonic and integer `atomicMin` == float min (R4). Clear value
= `floatBitsToUint(1.0)`.

Holes (no prev pixel maps to `p'`) and back-projection misses remain at the cleared
value `far`. 1-pixel cracks from forward scatter also remain `far` → safe.

Encoding: linear depth in `[0,1]` → `uint32` via `floatBitsToUint` (monotonic for
non-negative floats) or `round(d*0xFFFFFFFF)`. Clear value = encoding of `1.0`.

### Pyramid (Hi-Z) storage

Two resources (see R4):
- **Scatter target**: a full-res `uint` SSBO. Portable `atomicMin` on `floatBitsToUint`
  of the linear depth (monotonic for non-negative floats). Cleared to
  `floatBitsToUint(1.0)` each frame.
- **Pyramid**: a **half-float (R16F) mipmapped texture**, sampled by the project pass
  via `textureLod`. Built in a **single dispatch** (local-only SPD, R6): mip 0 is the
  uint scatter converted to float; mips 1..5 are max-reduced in shared memory per
  32×32 tile. No per-level dispatch, no atomics, no coherent memory.

This keeps the one mandatory atomic in an SSBO (portable on Metal) while the read side
is a simple float mip sampler. Fallback to R32F / all-uint if half precision is
insufficient on macOS.

### Project-pass cull (Step 3)

Insert the test in `ProjectGaussian()` **after** `center_pixel`, `bbox_half_extent`
and `norm_depth` are computed and after the existing screen-bbox cull, immediately
before packing `composite[...]` / returning `SortEntryParams`:

```glsl
if (OcclusionCullEnabled()) {
    float ext = max(bbox_half_extent.x, bbox_half_extent.y) * 2.0; // footprint px
    uint level = HiZLevelForExtent(ext);          // ceil(log2(ext)), clamped
    float prior_max = HiZSampleMaxLinear(center_pixel, level); // decode uint
    // norm_depth and prior_max both linear, near=0 far=1, larger=farther.
    if (norm_depth > prior_max + kOcclusionEpsilon) {
        return CulledEntryParams(splat_index);
    }
}
```

Single-tap at the level whose texel ≥ footprint is the standard conservative Hi-Z
read. Up to 4 taps is a later tightening refinement. `kOcclusionEpsilon` guards
quantization/temporal error (start ~1e-3 of the depth range; tune).

### Save merged depth (Step 1)

At the end of Stage B, store the merged **linear occluder** depth into the persistent
`prior_depth_tex`:
- `gaussian_composite.comp` emits a second, stricter-threshold occluder depth (R1).
- The extended `gaussian_depth_merge.comp` merges that occluder depth with the
  Filament scene depth (max-nearest) and writes persistent **linear** `prior_depth_tex`.
  Run it whenever the feature is on (not only on readback). When `has_scene_depth ==
  false`, skip the merge and use the composite occluder depth directly.

Also snapshot frame *N*'s `clip_from_world` (or its inverse) and near/far for next
frame's reproject pass (store CPU-side in the per-view resources / params).

---

## Resource and Code Changes

### New per-view GPU resources (`GaussianSplatViewGpuResources`, ComputeGPU.h)
- `prior_depth_tex` — R32F linear occluder depth of the previous frame (persistent).
- `reproj_scatter_buf` — uint SSBO, full-res scatter target for `atomicMin` (persistent;
  resized on viewport change).
- `hiz_pyramid_tex` — R16F mipmapped texture holding the max-depth pyramid (persistent).
- CPU-side: `prev_clip_from_world` (mat4), `prev_near`, `prev_far`, `prev_valid` flag.

### UBO change (`GaussianViewParams`, GaussianSplatDataPacking.h/.cpp)
Add, after the current 288 bytes:
- `mat4 prev_clip_from_world_inv` (offset 288) — unproject saved depth → world.
- `vec4 prev_depth_params` — prev near, far, `hiz_valid` flag, occlusion-enable flag.
- Pyramid level offset/stride table: pass as a small UBO/SSBO or push it into
  `limits`/a new `uvec4`; CPU computes per-level offsets.

`clip_from_world_{N+1}` is reconstructible from existing `view_from_world` ×
`clip_from_view`. Pack in `PackGaussianViewParams`; carry the previous frame's values
forward in the per-view resources.

### New shaders (ComputeGPU.h enum + `kGsShaderNames[]` + CMake SOURCES)
- `kGsDepthReproject` — **2×2-quad** forward scatter (R7): per quad, reproject 4
  corners, MAX-fill the interior when the reprojected bbox ≤ 3×3 else scatter corners
  only; writes the uint scatter SSBO with the pre-checked `atomicMin` + background-skip
  (R8).
- `kGsHiZReduce` — **single-dispatch** local-only SPD: one 32×32 tile per workgroup,
  converts the uint SSBO → R16F mip 0, then max-reduces mips 1..5 in shared memory
  (`subgroupMax`). No per-level loop, no atomics, no coherent memory. Update `kCount`.
- **Changed (no new shader):** `gaussian_composite.comp` gains the stricter occluder
  depth output; `gaussian_depth_merge.comp` is extended to write the persistent linear
  `prior_depth_tex` (the save step).

Register `.comp` files in
`cpp/open3d/visualization/rendering/gaussian_splat/CMakeLists.txt`
`open3d_add_compute_shaders(... SOURCES ...)`; add Vulkan binding tables
(`kBindings*`) and `kShaderBindings[]` entries in ComputeGPUVulkan.cpp, and the Metal
equivalents in ComputeGPUMetal.mm.

### Bindings
Free slots today: **9, 13, 17+** (Metal caps texture/sampler indices at 0..15;
SSBOs are less constrained). Proposed:
- Reproject pass: `0` UBO, saved-depth sampler/SSBO, pyramid-L0 SSBO (atomic).
- HiZ reduce pass: `0` UBO, pyramid SSBO (in/out via level offsets).
- Project pass: add pyramid SSBO at a free binding (e.g. **9** or **13**) — read-only.

### Pass-runner sequence (GaussianSplatPassRunner.cpp)
- **Stage A**, before PROJECT (and before its barrier): if `prev_valid` and enabled —
  (a) clear scatter SSBO to far, (b) `kGsDepthReproject`, barrier,
  (c) `kGsHiZReduce` (**single dispatch**), barrier. Then PROJECT (now reads pyramid).
- **Stage B**, after COMPOSITE: write/copy merged linear depth into `saved_depth_tex`;
  snapshot `prev_clip_from_world`/near/far; set `prev_valid = true`.

Mirror the same insertions in `FilamentRenderToBuffer` for offscreen renders.

---

## Implementation Steps (status log)

1. [ ] Add `RenderConfig` flag (e.g. `bool occlusion_cull = false`) + plumb to UBO.
2. [ ] Extend `GaussianViewParams` (prev matrices, prev near/far, flags, level table)
       and `PackGaussianViewParams`; carry prev-frame values in per-view resources.
3. [ ] Add `prior_depth_tex` + `reproj_scatter_buf` (uint SSBO) + `hiz_pyramid_tex`
       (R16F, mipmapped) to `GaussianSplatViewGpuResources`; allocate/resize in Vulkan
       and Metal backends.
4. [ ] `gaussian_composite.comp`: add stricter-threshold occluder depth output
       (`kOccluderThreshold`), second R32F image.
5. [ ] Extend `gaussian_depth_merge.comp`: write persistent **linear** `prior_depth_tex`
       (occluder ⊕ scene); run whenever feature on; skip/alias when no scene depth.
6. [ ] New `gaussian_depth_reproject.comp` (2×2-quad forward scatter, R7: MAX-fill
       when reprojected bbox ≤ 3×3 else corners-only; uint-SSBO `atomicMin` with
       background-skip + non-atomic pre-check, R8; linear-depth convention).
7. [ ] New `gaussian_hiz_reduce.comp` (single-dispatch local SPD: 32×32 tile/group,
       uint→float mip 0 convert, shared-memory `max` for mips 1..5; no atomics).
8. [ ] Register shaders (enum, names, CMake, Vulkan + Metal binding tables).
9. [ ] Wire passes into `GaussianSplatPassRunner` (Stage A reproject/reduce before
       PROJECT; Stage B save) and `FilamentRenderToBuffer`.
10. [ ] Add cull test in `gaussian_project.comp` (footprint-based single-tap Hi-Z read).
11. [ ] Tests: image-invariance (multi-pose, ON vs OFF) + effectiveness (entry-count
        drop) in `GaussianSplatRender.cpp`; optional Python parity if exposed.
12. [ ] Update `GaussianSplatDesign.md` (new passes, bindings, sort/UBO tables) and
        Doxygen/Sphinx docs; add a usage snippet for the new `RenderConfig` flag.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Temporal disocclusion (occluder moved/removed, fast camera) wrongly culls now-visible splats | Visible artifacts / popping | Conservative max-pyramid + holes-as-far; `kOcclusionEpsilon` guard band; multi-pose ON/OFF image-invariance test; default OFF |
| Depth-convention mismatch (inverse reversed-Z vs linear) | Mass over/under cull | Single documented contract; convert once in reproject; unit-style assert via effectiveness test |
| Forward-scatter cracks (fractional shift / magnification) silently disable culling for a whole coarse tile | Lost perf benefit (not incorrect) | **2×2-quad MAX-fill** (R7): fill compact (≤3×3) reprojected quads with the farthest corner depth; `round` not `floor`; cracks otherwise read as far (safe) |
| Filling cracks corrupts **real** thin scene gaps (grass, comb, chain-link) | Wrong image (background culled) | **MAX-fill is correct by construction** (R7): real gaps in a compact quad fill with the *background* (far) depth → not culled. Blind dilation / valid-children fill / half-res scatter **rejected** as anti-conservative |
| Float image-atomics / mipmapped-image portability (Vulkan drivers, Metal) | Build/runtime failures | **Resolved:** the one mandatory atomic (`atomicMin`) lives in a uint SSBO; the float/half pyramid is non-atomic store/sample. No texture atomics, no float-atomic extension |
| Net slowdown on low-occlusion scenes (reproject+reduce overhead > project savings) | Perf regression | Opt-in flag; document; possible heuristic to disable when prior cull ratio is low |
| Multi-object scenes share one merged depth, but per-object visibility toggles | Stale prior depth after `ShowGeometry` | `prev_valid=false` (skip a frame) on geometry add/remove/visibility change |
| Frame 0 / resize: no prior depth | Crash / wrong cull | `prev_valid` flag gates the cull; reset on resize/`InvalidateGaussianSplatOutput` |
| Metal first-frame deferred composite ordering | Save/reproject timing | Mirror the existing `#if defined(__APPLE__)` ordering used by composite |

## Open Decisions (to confirm before coding)
- `kOccluderThreshold` value (1/255 vs 1/4096) and `kOcclusionEpsilon` magnitude.
- Pyramid cap level (largest footprint: 32 vs 64 px).
- Binding indices for the pyramid sampler + scatter SSBO in the project pass (9 vs 13).
- The effectiveness-test entry-count drop threshold.
- Whether half (R16F) pyramid passes macOS testing or needs R32F fallback.

### Resolved by review
- Save step reuses `gaussian_depth_merge.comp` (no new save shader). ✔
- Pyramid storage: uint SSBO scatter + half-float mipmapped texture. ✔
- Reproject and HiZ-reduce stay separate passes/shaders (cannot be safely combined;
  merging into one shead file gives no benefit). ✔
- Pyramid built in a **single dispatch** via local-only SPD (32×32 tile/group, no
  atomics, no coherent memory; cross-tile SPD tail omitted thanks to footprint cap). ✔
- Project-pass read: single conservative tap (4-tap deferred). ✔
- **Crack mitigation: 2×2-quad MAX-fill, compactness cap `N = 3` fixed** (R7). Per-pixel
  dilation, valid-children fill, half-res scatter, and depth-continuity `τ` all
  **rejected** (anti-conservative or parameter-dependent). ✔
- **Scatter atomics: background-skip + non-atomic pre-check, always on** (R8).
  SLM-tiled local-reduce **deferred** (profiling-gated; benefits only minification and
  still needs global atomics on flush). ✔
- **Half-resolution scatter target rejected** (anti-conservative cross-source min); the
  correct-but-not-faster coarse alternative is sampling pyramid mip 1. ✔

---

## Performance Analysis (expectation, to validate)

Net win = (sort-entry + raster savings from culled splats) − (reproject + reduce + save
overhead). Overhead is fixed and screen-resolution-bound (~a few hundred µs); savings
scale with how much of the scene is occluded **and** correctly predicted from the prior
frame.

- **Smooth translation/pan/zoom, depth-complex scene:** ~15–35% lower frame time. The
  2×2-quad scatter (¼ invocations) + background-skip keep overhead low; most occluders
  reproject compactly.
- **Strong single foreground occluder (e.g. wall/object in front of dense field):**
  ~40%+, dominated by sort-cost elimination of fully-hidden splats.
- **Pure rotation:** parallax-free → reprojection is near-exact → close to best-case
  cull accuracy at low scatter cost.
- **Flat / sparse / front-facing scenes, or erratic fast motion:** ~0 to slightly
  negative (overhead not recovered) → hence **opt-in, default OFF**.

Distinct from existing mechanisms: the current scene-depth test culls only against
**mesh** depth at composite time; transmittance early-exit stops *accumulation* but
splats are still **sorted/projected**. This feature adds **GS-on-GS** occlusion at the
**project/sort** stage. Validate with `gs_counters.total_entries` (and frame time) ON
vs OFF across poses; require a measurable entry-count drop with byte-identical output
within tolerance.
