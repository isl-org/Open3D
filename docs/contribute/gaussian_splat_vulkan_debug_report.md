# Gaussian Splat Vulkan Interactive Debug Report

Last updated: 2026-09-04

## Status and locked scope

Goal: eliminate camera-motion flicker, freezes, and the eventual swapchain
presentation failure in interactive Gaussian Splat rendering on Windows with an
Intel Arc A770M.

Requirements:

- Preserve the passing source-Filament `RelWithDebInfo` build and
  `*SplatRender*` tests.
- Prefer Open3D-side fixes. Do not patch Filament unless an Open3D fix is
  impossible or has a significant runtime cost.
- Use host-side `engine_.flushAndWait()` synchronization where it is sufficient.
- Keep useful instrumentation and remove experiments for falsified hypotheses.
- Preserve unrelated worktree files.

Test method:

1. Build `GaussianSplat` and `tests` with `BUILD_FILAMENT_FROM_SOURCE=ON`.
2. Run `tests.exe --gtest_filter=*SplatRender*`.
3. Run the interactive viewer with `VK_LAYER_KHRONOS_validation` and CDB.
4. Exercise repeated dolly and drag using `gs_Skull.splat` on the Arc A770M.
5. Compare with mesh-only, GS-only, Linux Vulkan, and the Iris Xe where useful.
6. Run the controlled Linux scene and queue matrices defined below, using a
   real X11/XWayland swapchain and synchronization validation.

The goal, requirements, and test method are locked. Revise this section
explicitly before expanding scope.

Lock revision 2026-09-01: Linux changed from an informal comparison into a
controlled falsification campaign. The goal and requirements are unchanged.

## Observed failure

On Windows and the Intel Arc A770M, camera movement alternates between the 3DGS
result and the Filament mesh scene. The window eventually freezes and often
terminates with:

```text
Postcondition
in present:108
reason: Cannot present in swapchain.
FAST_FAIL_FATAL_APP_EXIT
```

The issue has not been observed on Linux Intel graphics or an NVIDIA dGPU.
Focused Arc tests pass:

```text
[==========] Running 2 tests from 1 test suite.
[  PASSED  ] 2 tests.
```

Full C++ tests and `OffscreenRendering` have also passed. The differentiating
surface is therefore the interactive frame, real window swapchain, shared image
state, or cross-command-system queue use.

## Interactive frame trace

The non-Apple frame sequence is:

1. `FilamentRenderer::BeginFrame()` calls `engine_.flushAndWait()` to drain the
   previous Filament frame before GS accesses a potentially shared queue.
2. `FilamentRenderer::BeginFrame()` starts asynchronous GS geometry work.
3. Filament `beginFrame()` begins its interactive frame and acquires a
   swapchain image when the first swapchain render pass starts.
4. `FilamentRenderer::Draw()` renders mesh scenes into the imported GS color
   and depth render target.
5. `engine_.flushAndWait()` completes Filament scene rendering.
6. GS waits for geometry, samples the shared scene depth, writes the shared
   color image, submits, and waits for composite completion.
7. Filament records the GUI/SceneWidget draw that samples the shared color.
8. Filament `endFrame()` submits and presents.
9. Camera input schedules the next redraw.

Relevant ownership boundaries:

- `GaussianSplatVulkanContext` owns the Vulkan instance, device, and GS queue.
- Filament receives the same device through `VulkanSharedContext`.
- GS and Filament use independent command pools and submission systems.
- Imported RGBA16F color and D32 depth images are visible to both systems.

## Queue topology

The native Arc A770M reports:

```text
family 0: queueCount=1, GRAPHICS | COMPUTE | TRANSFER | SPARSE, present=true
family 1: queueCount=4, COMPUTE, present=true
family 2: queueCount=2, TRANSFER | SPARSE, present=true
family 3: queueCount=2, VIDEO_DECODE, present=false
```

Current selection requires a graphics+compute family. It requests up to two
queues from that family, assigning queue 0 to GS and the last requested queue
to Filament. On Arc, family 0 has only one queue, so both receive the same
`VkQueue`:

```text
fam=0 gs_q=0 fil_q=0
```

### Why not use the compute-only family?

It is viable and may improve geometry-pass overlap, but it is not a free removal
of synchronization:

- Exclusive shared images require paired queue-family release/acquire
  ownership transfers.
- Concurrent image sharing across families 0 and 1 removes ownership transfers,
  at a possible driver-specific performance cost.
- Different queues still require a Vulkan memory dependency for
  graphics-write to compute-read and compute-write to fragment-read. Host fence
  waits establish host completion, but must not be assumed to establish the
  required cross-queue device memory dependency without specification and
  synchronization-validation evidence.
- The asynchronous GS geometry stage uses GS-private buffers and is the safest
  stage to move to family 1.
- Composite reads and writes Filament-shared images and is the difficult stage.

Candidate designs, in increasing complexity:

1. Serialize access to the single shared queue on Arc. This is the cheapest
   root-cause experiment, but limits geometry overlap.
2. Put GS geometry on family 1 and retain composite on family 0. This preserves
   shared-image simplicity but needs a hand-off for geometry output buffers.
3. Put all GS work on family 1, create shared images concurrently for families
   0 and 1, and use explicit cross-queue dependencies at the two image
   boundaries.
4. Use exclusive images with explicit ownership transfers and dependencies.
   This is most explicit and most complex.

Decision: do not change queue families until tracing establishes whether
concurrent host access to family 0 queue 0 occurs. If it does, test option 1 as
the cheapest discriminator, then evaluate option 2.

## Validation evidence

The latest CDB and validation run reports these defect classes in this order:

| Defect | First phase | Proven owner | Current interpretation |
|---|---|---|---|
| Image barrier `layerCount=0` | Startup | Filament attachment/render-target path | Real defect also emitted by working non-GS `Draw`; weak GS-specific evidence |
| Zero framebuffer/render area | Startup | Filament render-pass path | Real defect also emitted by working non-GS `Draw`; weak GS-specific evidence |
| Depth descriptor layout mismatch | Before semaphore errors | Filament graphics draw plus independently transitioned imported depth | Real shared layout-tracking defect |
| Graphics descriptor-set incompatibility | Before semaphore errors | Filament graphics pipeline/cache | Real Filament defect also emitted by working non-GS `Draw`; weak GS-specific evidence |
| Present semaphore reuse, VUID 00067 | Repeated interactive frames | Filament v1.54 command/present lifecycle | Real swapchain defect also emitted by working non-GS `Draw`; can explain an eventual present abort, not the GS-only flicker |
| `present()` postcondition failure | Terminal event | Filament/platform swapchain | Direct abort trigger |

Error ordering falsifies the claim that all earlier validation errors cascade
from present semaphore reuse. Several independent defects coexist.

### Windows diagnostic-mode audit

Audit date: 2026-09-01.

| Diagnostic mode | Installed | Confirmed active in a reproduction | Evidence |
|---|---|---|---|
| Khronos core validation | Yes, SDK 1.4.341.1 | Yes | Loader confirms instance and device insertion in seven CDB logs |
| `VK_EXT_debug_utils` GS labels | Extension available | Yes | Crash dump localizes incomplete work to `gs_composite` |
| Application debug-utils messenger | Extension available | No | Open3D enables the extension and command labels but does not create its own `VkDebugUtilsMessengerEXT` |
| Synchronization validation | Available through Khronos validation | Yes | Startup reports `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`; no `SYNC-HAZARD` was reported before the present abort |
| GPU-Assisted Validation (GPU-AV) | Available through Khronos validation | Yes | Startup reports `VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT`; no GPU-AV shader-access error was reported before the freeze |
| `VK_LAYER_LUNARG_crash_diagnostic` | Yes | Yes | Loader insertion confirmed; `cdl_dump.yaml` captured incomplete `gs_composite` work after a device error |
| CDB native exception debugging | Yes | Yes | CDB caught `FAST_FAIL_FATAL_APP_EXIT` in `ucrtbase!abort` after Filament's present postcondition |

The vkconfig installation contains templates mentioning GPU-AV and crash
diagnostics. Their presence is not evidence that either mode was selected for
an Open3D run.

Confirmed standard-validation CDB runs:

| Log | Validation errors | Present-semaphore VUID count |
|---|---:|---:|
| `arc_gs_depth_layout_cdb.log` | 54 | 20 |
| `arc_gs_interactive_cdb.log` | 54 | 20 |
| `arc_gs_per_image_semaphore_cdb.log` | 50 | 12 |
| `arc_gs_queue_idle_cdb.log` | 16 | 0 |
| `arc_gs_semaphore_quarantine_cdb.log` | 54 | 20 |
| `cdb-splat-after-vma.log` | 16 | 0 |
| `cdb-splat.log` | 0 | 0 |

### Working non-GS `Draw` control

Audit date: 2026-09-02. The visually correct `Draw` example was run through its
first interactive window for bounded 12-15 second captures. A timeout is
expected because `Draw` waits for each of four interactive windows to close.
The application continued rendering during every capture.

| Mode | Validation error callbacks | Unique VUIDs | Distinct finding |
|---|---:|---:|---|
| Khronos core | 88 | 8 | None beyond the shared core errors below |
| Synchronization validation | 4,662 | 9 | No `SYNC-HAZARD`; the ninth VUID was framebuffer flag/imageless state `04533` |
| GPU-AV | 1,110 | 8 | No GPU-AV shader-access or descriptor-runtime finding |
| Best practices plus thread safety | 18 | 7 | No best-practices or thread-safety diagnostic |

The callback totals depend on frame rate and duplicate-message limits; the
presence and type of each finding are the useful comparison. Core, syncval,
and GPU-AV all reproduced these eight VUIDs in the working control:

- `VUID-vkQueueSubmit-pSignalSemaphores-00067`
- `VUID-VkImageSubresourceRange-layerCount-01721`
- `VUID-vkCmdDrawIndexed-None-08600`
- `VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358`
- `VUID-VkRenderPassBeginInfo-pNext-02853`
- `VUID-VkRenderPassBeginInfo-None-08996`
- `VUID-VkFramebufferCreateInfo-width-00885`
- `VUID-VkFramebufferCreateInfo-height-00887`

These are real Vulkan or Filament defects, but they are non-discriminating for
the GS-only flicker because correct non-GS rendering emits them too. They should
not drive the primary GS investigation unless a controlled run shows that GS
changes their timing or consequence. In contrast,
`VUID-vkCmdDraw-None-09600`, the imported-depth descriptor layout mismatch,
did not occur in any `Draw` mode and remains GS-specific evidence. The crash
diagnostic localization to incomplete `gs_composite` work is also GS-specific.

Repeating a validation mode without better labels or scene controls now has low
diagnostic value. Future comparisons should use the intersection with this
working control to separate baseline errors from GS-only findings.

#### What each mode can falsify

**Khronos core validation plus debug utils**

Can identify or substantially falsify under an equivalent stress path:

- invalid image layouts and usage;
- zero attachment layers and zero framebuffer/render areas;
- incompatible descriptor sets and pipeline layouts;
- binary semaphore reuse and many object-lifetime violations;
- invalid swapchain acquire/present API usage.

It cannot reliably falsify:

- concurrent host calls on the same `VkQueue`;
- a valid-API driver bug, TDR, or GPU firmware issue;
- application stage skipping or presentation of a stale but valid image;
- a CPU deadlock;
- shader accesses that are API-valid but out of bounds at runtime.

Enabling `OPEN3D_VULKAN_DEBUG_UTILS=1` adds GS command labels to validation and
crash tools. For an application callback, create and register a
`VkDebugUtilsMessengerEXT`; merely enabling the extension is not equivalent to
registering a messenger. The existing layer stderr output already proves that
core validation was active.

**GPU-Assisted Validation**

GPU-AV was confirmed active on the Arc reproduction. It reported no dynamic
buffer/image bounds error, invalid descriptor access, descriptor-index error,
or invalid dispatch data before the viewer became unresponsive. It repeated
the known core errors. This lowers, but does not eliminate, a GS shader memory
fault because GPU-AV changes shader code, memory use, and timing.

A clean GPU-AV mixed stress run lowers the probability of a GS shader memory
fault, especially when GS-only and mixed modes both execute the same dispatch
sizes. It does not falsify queue host synchronization, present semaphore
lifetime, WSI resize behavior, stage skipping, or a CPU deadlock. GPU-AV changes
shader code, memory use, and timing, so a clean run cannot completely exclude a
timing-sensitive driver fault.

**LunarG crash diagnostic layer**

This layer produced a dump 6.662 seconds after startup, before its 30-second
watchdog could fire. It reported a device error and queue 0 stalled with
`completedSeq=3157`, `submittedSeq=3160`. The incomplete command buffer was
labeled `gs_composite`: its label begin completed, while push descriptors,
`vkCmdDispatch(1024, 1, 1)`, the following barrier, and command-buffer end were
incomplete. This proves a genuine device failure occurs in addition to the
later Filament present abort.

The Arc driver exposes neither `VK_EXT_device_fault` nor
`VK_AMD_buffer_marker`, so the layer could not provide a vendor fault address
and disabled semaphore tracking. The dump localizes the stalled work but does
not prove whether the dispatch, invalid shared-image state, same-queue host
access, or the Arc driver caused the device failure.

Crash diagnostics can identify the last completed GPU work and provide useful
device-fault context for a real GPU hang, TDR, or `VK_ERROR_DEVICE_LOST`. A dump
that identifies a GS dispatch can support a shader/driver-hang hypothesis. A
dump that ends in unrelated Filament or presentation work shifts attribution.
Absence of a dump does not falsify these causes when the application merely
aborts on a non-device-lost present error.

Before relying on this layer, log the numeric `VkResult` from acquire, submit,
and present. This distinguishes `VK_ERROR_DEVICE_LOST`,
`VK_ERROR_OUT_OF_DATE_KHR`, surface loss, and other WSI failures.

## Hypothesis matrix

### Falsified or rejected

| Hypothesis or attempted fix | Evidence | Action |
|---|---|---|
| GPU-AV-detectable GS shader memory fault in this run | GPU-AV was active and emitted no shader-access finding before the freeze | Lowered, not globally excluded; investigate only with dispatch isolation or new GPU-AV evidence |
| Wrong physical device or different logical devices | Both systems report the Arc shared device | Closed |
| Different current queue families | Both use family 0, queue 0 | Closed |
| Missing swapchain acquire/resize patch | Tracked patch 0002 is applied | Closed |
| Post a redraw after every non-Apple composite | Creates an endless redraw chain during `OnDraw()` | Reverted |
| Add `vkQueueWaitIdle()` to Filament's regular wait path | Produced a blank/frozen viewer | Reverted |
| Restore imported depth to depth-attachment layout after GS composite | Visible behavior unchanged; later Filament draw expected GENERAL or shader-read | Reverted 2026-09-01 |
| Replace only the semaphore passed to `present()` | Submit had already selected and signaled the old semaphore | Reverted |
| Treat all validation errors as a semaphore cascade | Earlier errors precede first semaphore VUID | Rejected |

### Viable causes

Cost scale: **low** is a local instrumentation/edit plus one existing build or
run; **medium** needs new controls, several interactive runs, or changes across
an ownership boundary; **high** needs a Filament/queue redesign, broad
cross-platform validation, or external driver investigation. Estimates include
the expected validation work, not just edit size.

| Cause | Current evidence and status | Next discriminator | Test cost | Potential solution | Fix cost |
|---|---|---|---|---|---|
| Device failure during GS composite | **Confirmed event, cause unresolved.** Two crash dumps stop at the same labeled composite command buffer; the second includes the pre-geometry wait | Add GS-only/mixed controls, then isolate the composite dispatch and its three image inputs | **Medium:** example controls, repeated Arc runs, crash-dump comparison | Fix the invalid composite input/state if isolated; otherwise produce an Arc driver reproducer | **Medium-high:** local if one resource is wrong; high/external if driver-specific |
| Remaining concurrent host access to the same `VkQueue` | **Lowered.** The known cross-frame gap is fixed by the retained pre-geometry wait, but the device failure is unchanged | Trace thread ID and begin/end around every GS and Filament queue submit/present to find any remaining overlap | **Medium:** Open3D tracing plus temporary generated-Filament tracing and one Arc run | Serialize all calls to the shared queue, or split selected GS work onto compute family 1 | **Medium-high:** locking is medium; split-family ownership and dependencies are high |
| Incorrect present semaphore lifetime | **Confirmed baseline defect, downgraded for GS flicker.** Working `Draw` repeatedly emits VUID 00067. It can explain an eventual present abort but does not discriminate the GS-only visual failure | Compare present results and timing in mesh-only, GS-only, and mixed controls only if the terminal abort remains a separate target | **Medium:** temporary Filament instrumentation and scene controls | Use per-swapchain-image semaphores or present fences; otherwise backport corrected Filament ownership | **High:** Filament lifecycle change plus WSI testing across platforms |
| Independent imported-depth layout trackers | **Confirmed GS-specific defect; high priority.** VUID 09600 occurs on the mixed imported-depth path and was absent from every working `Draw` control | Trace both trackers and run mixed versus GS-only with depth sampling disabled as an isolation probe | **Medium:** image tracing and controlled Arc runs | Define an explicit hand-off layout and update both trackers, or assign transition ownership to one system | **Medium-high:** crosses Open3D/Filament imported-image state |
| Shared-color/depth state invalid during composite | **Viable and high priority.** It fits the stalled composite and alternating output; syncval found no generic hazard, while the GS-only depth VUID proves one invalid imported-image state | Replace each composite image input with a known-valid temporary resource one at a time and compare crash dumps | **Medium:** three local isolation probes and repeated Arc runs | Add correct hand-off barriers/layouts and consistent imported-image tracking | **Medium-high:** local barriers may be medium; shared tracker changes are high |
| Filament graphics descriptor-cache incompatibility | **Confirmed baseline defect, downgraded.** Working `Draw` emits the same VUIDs while rendering correctly, and they do not explain a compute dispatch stall | Investigate separately only if a GS control changes the error or visibly loses mesh/UI output | **Medium:** controls plus temporary Filament tracing | Correct cache invalidation/keying or backport the focused upstream fix | **Medium-high:** Filament-side fix and graphics regression coverage |
| Zero viewport/render target | **Confirmed baseline defect, downgraded.** Working `Draw` emits the same errors while rendering correctly | Investigate separately if GS-only tracing ties a zero-sized target to the failing composite frame | **Low:** local logging and one validation run | Skip zero-sized passes and defer target creation until dimensions are valid | **Low-medium:** small edit; resize/startup regression tests required |
| Zero attachment layer count | **Confirmed baseline defect, downgraded.** Working `Draw` emits the same error while rendering correctly | Investigate separately if the affected image handle is one of the failing composite inputs | **Low:** local probe and one validation run | Initialize and preserve `layerCount=1` for the affected attachment | **Low:** local initialization fix and focused validation |
| Failed stage presents partial output | **Viable visual explanation, not a device-failure explanation.** It directly fits alternating GS/mesh generations | Trace frame/generation IDs and stage results through GUI sampling and present | **Medium:** per-frame Open3D tracing and motion reproduction | Present only a completed generation and retain the last complete output on failure | **Medium:** renderer state change and interactive regression tests |
| Resize/recreation or imported-image lifetime | **Strengthened.** Validation observed a recreated `1024x764` swapchain used by a `1024x768` framebuffer | Add deterministic resize stress and trace extents plus wrapper/native image lifetimes | **Medium:** resize control, tracing, and repeated validation runs | Recreate dependent targets atomically, reject stale extents, and serialize destruction | **Medium-high:** multi-object lifetime change across GUI and renderer |
| Arc driver/TDR/hybrid presentation behavior | **Viable residual cause.** Device failure is Arc-specific, but known Vulkan-invalid state remains | Reproduce after all relevant VUIDs are removed; compare Arc driver versions and Iris/Linux matched paths | **High/external:** multiple systems or driver installations and long stress runs | Driver update/rollback, workload mitigation, or minimal vendor report | **External/high:** may not be fixable in Open3D |

## Detection matrix

| Cause | Linux Vulkan | This Windows Arc machine |
|---|---|---|
| Same-queue host race | May reproduce as corruption/freeze, but validation usually cannot detect host `VkQueue` races | Easy to trace; cheap forced-serialization A/B test |
| Present semaphore reuse | Validation should report VUID 00067 with a real window swapchain; headless tests will not | Already identified reliably by validation |
| Depth layout mismatch | Validation should report regardless of vendor if the same path executes | Already identified reliably; debug labels will locate the image |
| Color layout/visibility mismatch | Synchronization validation may report it; ordinary validation may not | Syncval was active and reported no hazard; image-state isolation and layout tracing remain |
| Descriptor incompatibility | Ordinary validation should report deterministically on Linux too | Already identified reliably |
| Zero framebuffer/viewport | Ordinary validation reports it on any Vulkan implementation | Already identified reliably |
| Zero attachment layer count | Ordinary validation reports it on any Vulkan implementation | Already identified reliably |
| Stage skipping/stale output | Logging exposes it on every platform; may not visibly flicker on tolerant drivers | Easy with per-frame generation logging |
| Resize/lifetime error | Validation or direct failure if exercised; compositor differences affect reproducibility | Easy to trace under CDB and validation |
| Arc driver/TDR | Not expected to reproduce on non-Arc Linux hardware | Windows event/TDR data and Arc/Iris comparison only |

Linux tests must use a real interactive window. Existing offscreen/headless tests
cannot expose presentation semaphore lifetime or window resize behavior.

## Filament patch audit

As of 2026-09-01:

- `3rdparty/filament` is a regular tracked subtree and is clean.
- Generated Vulkan files under `build/filament/src/ext_filament` match the
  corresponding `3rdparty/filament` files.
- No generated-source diagnostic or semaphore experiment remains.
- No uncommitted Filament patch exists.
- The only tracked patch files are:
  - `0001-importTextureR.patch`
  - `0002-handle-vulkan-swapchain-acquire.patch`
- Patch 0002 includes `vkQueueWaitIdle()` only during swapchain destruction, not
  in the per-frame render path.

Do not add another Filament patch until an Open3D-side experiment demonstrates
that a Filament change is necessary and lower-cost alternatives are excluded.

## Instrumentation plan

Add opt-in tracing controlled by an environment variable. Each line should
include a monotonic sequence number, frame number, thread ID, queue handle,
view pointer, and relevant Vulkan handles.

Trace points:

1. Camera redraw request and `Window::OnDraw()` begin/end.
2. `FilamentRenderer` begin, scene flush begin/end, GS composite begin/end, GUI
   draw, and end frame.
3. GS command-buffer begin, geometry submit, geometry fence completion,
   composite submit, and composite fence completion.
4. Shared color/depth registration, old/new layouts, allocation, and release.
5. Filament swapchain acquire image/result, submission semaphore, present
   image/semaphore/result, and recreate.
6. Per-view viewport, dirty flags, geometry/composite result, and output
   generation.

Filament instrumentation should first be a generated-source diagnostic probe.
Only convert a successful probe into a tracked patch if it must remain.

## Next discriminating tests

1. Add GS-only, mesh-only, and mixed scene controls and reproduce each under
   crash diagnostics and core validation.
2. Isolate the composite image inputs one at a time to identify which resource
   or state is required for the Arc device failure.
3. Add Open3D per-frame, shared-image, and queue tracing without changing
   behavior.
4. Instrument Filament submit/acquire/present in generated source only.
5. Use the resulting evidence to choose between shared-image state repair,
   remaining queue serialization, and an Arc-specific composite reproducer.
6. Track the working-`Draw` baseline defects separately; do not block the GS
   investigation on zero-size, zero-layer, graphics-descriptor, or present-
   semaphore fixes unless a control demonstrates GS-specific impact.

## Linux falsification campaign

Linux is a known-good visual baseline, not automatically a falsification host.
A Linux pass falsifies a candidate only when the relevant code path, queue
topology, workload, and Vulkan validation conditions match. If those differ, a
pass establishes platform dependence or narrows the trigger rather than
falsifying the Windows hypothesis.

### Host and display requirements

Use a physical Vulkan GPU and a real X11 or XWayland window. This branch creates
an Xlib Vulkan surface, so `DISPLAY` must be set even in a Wayland session. Do
not use SSH X forwarding or Xvfb unless `vulkaninfo` confirms that the intended
hardware driver and presentation surface are still selected.

Record the environment before testing:

```bash
git rev-parse HEAD
echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_SESSION_TYPE=$XDG_SESSION_TYPE"
vulkaninfo --summary | tee linux-vulkan-summary.log
vulkaninfo 2>&1 | sed -n '/VkQueueFamilyProperties:/,/VkPhysicalDeviceMemoryProperties:/p' \
      | tee linux-vulkan-queues.log
```

The Open3D log must identify the expected physical device and report the queue
assignment:

```text
GaussianSplat VulkanContext: ready '<device>' (fam=<family> gs_q=0 fil_q=<index>)
```

### Comparable build

Use the same source Filament and a non-headless GUI build:

```bash
cmake -S . -B build-linux -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_FILAMENT_FROM_SOURCE=ON \
      -DBUILD_GUI=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_CUDA_MODULE=OFF \
      -DBUILD_SYCL_MODULE=OFF \
      -DBUILD_WEBRTC=OFF
cmake --build build-linux --parallel "$(nproc)" --target GaussianSplat Draw tests
```

Do not compare a precompiled newer Filament against the Windows source-built
v1.54 tree; that would confound platform and dependency version.

### Validation configuration

Install `vulkan-tools` and `vulkan-validationlayers` (package names on Ubuntu
and Debian), then create an untracked settings directory:

```bash
mkdir -p build-linux/validation
cat > build-linux/validation/vk_layer_settings.txt <<'EOF'
khronos_validation.validate_sync = true
khronos_validation.syncval_submit_time_validation = true
khronos_validation.syncval_shader_accesses_heuristic = true
khronos_validation.duplicate_message_limit = 1000
khronos_validation.report_flags = error,warn
EOF
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_SETTINGS_PATH="$PWD/build-linux/validation"
export OPEN3D_VULKAN_DEBUG_UTILS=1
export VK_LOADER_DEBUG=error,warn
```

Confirm at startup that `VK_LAYER_KHRONOS_validation` was loaded. Do not rely
on the nonstandard `VK_LAYER_KHRONOS_validation_SYNC` environment variable.

Capture each run independently while preserving its exit status:

```bash
set -o pipefail
build-linux/bin/examples/GaussianSplat /data/gs_Skull.splat 2 0 0 \
      2>&1 | tee linux-mixed-validation.log
status=${PIPESTATUS[0]}
echo "viewer_exit=$status" | tee -a linux-mixed-validation.log
```

### Required example controls

The current `GaussianSplat` example always creates both the input splat and a
red cube. Before the controlled campaign, add example-only diagnostic options
that do not change renderer behavior:

- `--scene=mixed|gs|mesh`, defaulting to `mixed`;
- `--stress-frames=N` for deterministic continuous camera motion and redraw;
- a fixed camera trajectory repeated for the requested frame count;
- per-frame sequence and camera-generation logging when tracing is enabled;
- `OPEN3D_GS_FORCE_SINGLE_QUEUE=1` to assign both systems queue 0 when the
   selected graphics+compute family exposes multiple queues;
- `OPEN3D_GS_SERIALIZE_BEFORE_GEOMETRY=1` as a temporary A/B discriminator.

The scene switch is preferable to using unrelated viewers because all three
controls then use the same engine, window, camera, GUI, swapchain, target
creation, and redraw code.

### Baseline run matrix

Run every row long enough to complete at least 10,000 presented stress frames.
Also perform maximize/restore and continuous resize cycles in a separate run.

| Run | Scene | Motion | Purpose |
|---|---|---|---|
| L0 | Mesh only | Static then stress | Generic Filament, GUI, descriptor, framebuffer, and present behavior |
| L1 | GS only | Static then stress | GS compute, imported color, stage completion, and present behavior without mesh occlusion |
| L2 | Mixed | Static | Shared color/depth initialization without camera churn |
| L3 | Mixed | Stress | Exact known-good counterpart to the Windows failure |
| L4 | Mixed | Stress plus resize | Swapchain recreation and imported-target lifetime |

For each log, count validation errors rather than relying on visual success:

```bash
rg -o 'VUID-[A-Za-z0-9-]+' linux-*.log | sort | uniq -c
rg -n 'Cannot present|device lost|FAILED|GaussianSplat VulkanContext: ready' linux-*.log
```

Interpret the scene controls as follows:

- Error in L0: generic Filament/window/swapchain issue; GS is not required.
- Error absent in L0 but present in L1: GS compute, imported color, or frame
   scheduling issue; mesh depth is not required.
- Error absent in L1 but present in L2/L3: imported depth or mixed-target
   hand-off issue.
- Error only in L3/L4: repeated-frame scheduling, queue overlap, presentation,
   or resize lifetime issue.
- VUID present in Linux without visual failure: the defect is cross-platform
   but is not sufficient by itself to cause the Arc symptom.

### Queue and scheduling matrix

First record the native queue topology. If Linux naturally reports
`gs_q != fil_q`, its visual success says nothing decisive about Arc's shared
queue. Run this matrix on a Linux GPU whose graphics+compute family exposes at
least two queues:

| Run | Queue assignment | Geometry scheduling | Discriminator |
|---|---|---|---|
| Q0 | Native separate queues | Asynchronous | Known-good baseline |
| Q1 | Forced shared queue 0 | Asynchronous | Reproduce possible concurrent host queue access |
| Q2 | Forced shared queue 0 | Serialized before geometry | Remove host overlap while keeping queue topology |
| Q3 | Native separate queues | Serialized before geometry | Measure serialization-only behavior and cost |

Collect queue handle, thread ID, monotonic sequence, and timestamps around all
GS submits and Filament submit/present operations.

Conclusions:

- Q1 fails and Q2 passes: strong evidence for shared-queue host contention.
- Q1 logs overlapping host calls even if rendering looks correct: the Vulkan
   external-synchronization defect is confirmed but not sufficient on Linux.
- Q1 passes with no overlapping queue calls: same-queue overlap is falsified
   for this schedule, not for Windows.
- All four pass with identical traces: lowers the priority of queue topology
   and shifts attention to Windows WSI/driver behavior.
- Q0 fails validation: fix the cross-platform violation before interpreting
   Arc-specific behavior.

### Hypothesis-by-hypothesis conclusions

| Candidate | Linux evidence that is useful | What Linux cannot prove |
|---|---|---|
| Shared-queue host race | Q1/Q2 plus timestamped queue-call overlap can confirm or falsify overlap under the tested schedule | A clean run on separate queues does not test it; a clean shared-queue run does not exclude an Arc-driver sensitivity |
| Present semaphore lifetime | VUID 00067 under L0 proves GS is unnecessary; under only L1-L4 it shows frame scheduling affects reuse | Absence does not prove v1.54 ownership is valid for every WSI image order |
| Depth layout tracking | Error only in mixed L2-L4 strongly isolates depth hand-off | A clean Linux driver path does not falsify a Windows-specific tracked-layout sequence |
| Color layout/visibility | Synchronization VUID in L1 isolates imported color; generation logs detect stale frames | Visual correctness alone cannot prove Vulkan memory dependencies are valid |
| Descriptor incompatibility | L0 reproduction proves generic Filament graphics state; absence from all exact-path runs makes it Windows-path-specific | It does not by itself identify the cache-key producer |
| Zero viewport/layer count | L0-L2 startup VUIDs localize the first view/target that creates them | Their absence only shows different startup topology unless traces match |
| Partial/stale output | Frame-generation logs can prove every presented frame completed geometry and composite | Visual inspection alone is insufficient |
| Resize/lifetime | L4 validation plus handle lifecycle logs can reproduce or substantially lower its priority | Different X11/Windows WSI recreation behavior remains |
| Arc driver/TDR | Cross-platform clean traces after all matching controls increase its relative likelihood | Linux cannot falsify a Windows Arc driver defect |

### Native debugger and sanitizer follow-up

If Linux aborts or freezes, run the exact failing control under GDB:

```bash
gdb --args build-linux/bin/examples/GaussianSplat /data/gs_Skull.splat 2 0 0
(gdb) catch signal SIGABRT
(gdb) run
(gdb) thread apply all bt full
```

Use a separate AddressSanitizer build only if handle lifecycle or use-after-free
remains viable. ThreadSanitizer can find C++ data races but is not expected to
understand Vulkan queue external-synchronization requirements.

### Campaign exit criteria

The Linux campaign is complete when:

1. L0-L4 and Q0-Q3 have logs, exit status, GPU/driver identity, and queue
    topology recorded.
2. Every Windows VUID has been classified as reproduced, absent on an
    equivalent path, or not exercised.
3. Per-frame traces establish whether mixed motion ever presents an incomplete
    GS generation.
4. Queue traces establish whether shared-queue host calls overlap.
5. Visual success is reported separately from Vulkan-valid execution.

## Progress log

- 2026-09-04: Implemented an Open3D-side no-copy layout hand-off for imported
   Vulkan render targets. Imported RGBA16F color now follows Filament's actual
   `VK_IMAGE_LAYOUT_GENERAL` policy, with `GENERAL`-to-`GENERAL` memory barriers
   around compute access. Imported D32 depth is transitioned from
   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL` to
   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` for exact `texelFetch()` reads and
   restored to the Filament-tracked attachment layout before composite
   submission completes. Filament's import patch no longer seeds new color
   images as already `GENERAL`, allowing Filament to perform the real initial
   transition from `UNDEFINED`. `GaussianSplat` and `tests` build in
   `RelWithDebInfo`; all three `*GaussianSplatRender*` tests pass on the B580.
   An interactive `hornedlizard.spz` run displayed correctly on the B580,
   replacing the previously blank/unresponsive result. The same run still
   emitted VUID 09600 from Open3D's `vkQueueSubmit2`: the composite descriptor
   expected the imported D32 image in
   `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, but validation observed
   `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`. Investigation found that
   `RunGaussianCompositePass()` calls `FinishGpuWork()`, ending and submitting
   the Vulkan command buffer before the `GpuComputeFrame` destructor calls
   `EndCompositePass()`. The latter consequently attempts to record the depth
   restore after submission and updates Open3D's software tracker even though
   no valid restore command was recorded. Replaced `FinishGpuWork()` with
   `frame.End()` on the normal path. This records the restore while the command
   buffer is active and then submits/waits exactly once; RAII still closes
   early-return paths. The `RelWithDebInfo` `GaussianSplat` and `tests` targets
   rebuilt successfully, and all three `*GaussianSplatRender*` tests pass on the
   B580. A subsequent interactive validation run still reported VUID 09600 with
   the sampled-depth descriptor expecting `SHADER_READ_ONLY_OPTIMAL` while the
   actual image was `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`. The sequencing bug was
   real, but fixing it did not remove the validation error and is not a complete
   shared-depth ownership solution. Interactive and offscreen rendering are
   visually correct; investigation of the remaining VUID and long-running
   freeze/device failure is paused at this checkpoint.
- 2026-09-04: Reproduced the Windows failure on an Intel Arc B580 using a
   source-Filament `RelWithDebInfo` build (`BUILD_FILAMENT_FROM_SOURCE=ON`) and
   deployed `Open3D.dll` and `tbb12.dll` beside `GaussianSplat.exe`. CDB plus
   `VK_LAYER_KHRONOS_validation` and `VK_LAYER_LUNARG_crash_diagnostic` reached
   the expected Vulkan device (`fam=0 gs_q=0 fil_q=0`). The focused
   `*GaussianSplatRender*` tests passed 3/3, matching the report's distinction
   between passing offscreen tests and the failing interactive Windows path.
   The first pre-source-Filament run crashed in
   `VulkanTexture::getLayout()` with a null `RangeMap` owner; this was not a
   valid comparison and was superseded by the source-Filament run.
- 2026-09-04: The comparable B580 run reproduced the GS-specific imported
   depth layout error (`VUID-vkCmdDraw-None-09600`), generic Filament layer/
   framebuffer/descriptor errors, and present semaphore reuse
   (`VUID-vkQueueSubmit-pSignalSemaphores-00067`). GPU-assisted validation also
   reported a shared-memory warning for `s_stolen_tile` in
   `gaussian_composite.comp`, but the shader already has a workgroup barrier
   immediately after the thread-0 write, so this is not treated as a confirmed
   shader defect. Crash diagnostics recorded the B580 queue at
   `completedSeq=372`, `submittedSeq=372`, with no command buffers listed after
   the watchdog, indicating a device/driver stall rather than a CPU exception.
- 2026-09-04: Tested and reverted an Open3D-side A/B experiment that seeded
   newly created imported images as `VK_IMAGE_LAYOUT_UNDEFINED` instead of the
   assumed attachment layouts. It removed the first depth-layout report but a
   later depth mismatch, semaphore reuse, and device stall remained. The
   experiment passed the focused tests but did not resolve the interactive
   failure, so no speculative tracker change is retained.
- 2026-09-02: Ran the visually correct non-GS `Draw` example under core,
  synchronization, GPU-assisted, and best-practices plus thread-safety
  validation. It reproduced eight core VUIDs previously seen in the GS run,
  with no `SYNC-HAZARD`, GPU-AV runtime finding, or thread-safety diagnostic.
  Those shared errors remain real defects but are downgraded as explanations
  for GS-only flicker. Depth-layout VUID 09600 and incomplete `gs_composite`
  crash localization remain GS-specific evidence.
- 2026-09-01: Added `engine_.flushAndWait()` before GS geometry submission.
   `GaussianSplat` builds and focused `*SplatRender*` tests pass 2/2.
- 2026-09-01: Repeated the crash-diagnostic run with the pre-geometry wait. A
   device error recurred after 4.345 seconds. The dump again stopped in
   `gs_composite`, with the label begin complete and push descriptors,
   `vkCmdDispatch(1024, 1, 1)`, and the following barrier incomplete. The wait
   closes the cross-frame queue-host gap but does not resolve the device failure.
- 2026-09-01: Confirmed GPU-AV active on Arc. It found no GPU-AV-specific
   shader-access error before the viewer froze; known core VUIDs remained.
- 2026-09-01: Confirmed synchronization validation active with submit-time and
   shader-access checks. It reported no `SYNC-HAZARD` before the same present
   abort, lowering detectable missing-memory-dependency hypotheses.
- 2026-09-01: Captured a crash diagnostic dump after a device error. Queue 0
   stopped at sequence 3157/3160 in `gs_composite`; push descriptors, dispatch,
   and the following barrier were incomplete. Arc lacks device-fault details.
- 2026-09-01: GPU-AV also exposed a recreated `1024x764` swapchain used by a
   `1024x768` framebuffer, adding concrete resize/extent evidence.
- 2026-09-01: Removed the falsified depth-restoration experiment.
- 2026-09-01: Rebuilt `GaussianSplat` and `tests`; focused Arc tests pass 2/2.
- 2026-09-01: Audited Filament trees; no uncommitted or generated-only patch
  remains.
- 2026-09-01: Captured Arc queue topology. Dedicated compute family 1 is
  available, while graphics family 0 exposes only one queue.
- 2026-09-01: Separated validation defects by first occurrence; rejected the
  previous single-cascade explanation.
- 2026-09-01: Added the controlled Linux scene, queue, validation, and stress
   matrices to distinguish visual success from Vulkan-valid execution.
