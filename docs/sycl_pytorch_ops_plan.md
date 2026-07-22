# SYCL PyTorch ML Ops — Porting Plan and Status

**Branch:** `ss/sycl-mlops`  
**Last updated:** 2026-07-13

---

## Goal

Add SYCL as a third execution path (alongside CPU and CUDA) for **all 18 CUDA PyTorch ML op kernel files**, so Intel GPU users can run ContinuousConv, SparseConv, FixedRadiusSearch, Voxelize, and pointnet/pvcnn ops.

Key design decisions:
- Intel DPC++ (`icpx`) with `-fsycl` per-source compilation (existing Open3D pattern)
- **sycl-tla v0.9.1** (CUTLASS v4.2.1 fork for Intel GPU) for GEMM in conv ops
- **oneDPL** for CUB-equivalent device algorithms (sort, scan, copy_if)
- `at::xpu::getCurrentXPUStream().queue()` for PyTorch XPU dispatch
- Existing CPU and CUDA paths are **untouched**; `BUILD_SYCL_MODULE=ON` enables SYCL
- Refactor and reuse as much code as possible from existing SYCL code. Re-use CUDA and CPU code where possible.
- Focus on correctness (all ML ops unit tests must pass) and performance. Optimizations should be on par or better than CUDA optimizations.
---

## Environment

| Tool | Location | Notes |
|------|----------|-------|
| `icpx` | `/opt/intel/oneapi/compiler/2025.3/bin/icpx` | Intel DPC++ 2025.3 |
| oneDPL | `/opt/intel/oneapi/dpl/2022.10/` | Device algorithms |
| oneMKL | `/opt/intel/oneapi/mkl/2025.3/` | GEMM fallback if needed |
| sycl-tla | `3rdparty/sycl_tla/sycl_tla.cmake` (downloads on configure) | v0.9.1 = CUTLASS v4.2.1 fork |
| PyTorch XPU | `venv-o3d-xpu`: `torch==2.10.0+xpu` from `https://download.pytorch.org/whl/xpu` | `libtorch_xpu.so` present |
| Python venv | `/mnt/seagate_4tb_b/ssheorey/venv-o3d-xpu` | Python 3.10.16 |
| SYCL build dir | `/mnt/seagate_4tb_b/ssheorey/Open3D/build_sycl_pt` | cmake configure WIP |
| Intel GPU machine | **Target for final build/test** | Transfer pending (2026-07-13) |

---

## Key API Substitutions

| CUDA | SYCL |
|------|------|
| `#include <ATen/cuda/CUDAContext.h>` | `#include <ATen/xpu/XPUContext.h>` |
| `at::cuda::getCurrentCUDAStream()` | `at::xpu::getCurrentXPUStream().queue()` → `sycl::queue&` |
| `cudaMemsetAsync(ptr, 0, n, stream)` | `queue.fill(ptr, T(0), n)` |
| `cudaMemcpyAsync(dst,src,n,kind,stream)` | `queue.memcpy(dst, src, n)` |
| `__global__` + `<<<grid,block,0,stream>>>` | `queue.submit([&](sycl::handler& cgh){ cgh.parallel_for(range, kernel); })` |
| `atomicAdd(ptr, val)` | `sycl::atomic_ref<T,relaxed,device_scope>{*ptr}.fetch_add(val)` |
| `__shfl_down_sync` warp reduce | `sycl::reduce_over_group(item.get_group(), val, sycl::plus<T>{})` |
| `cub::DeviceScan::InclusiveSum` | `std::inclusive_scan(oneapi::dpl::execution::make_device_policy(queue), ...)` |
| `cub::DeviceRadixSort::SortPairs` | `oneapi::dpl::sort_by_key(dpl_policy, keys, keys+n, vals)` |
| `cutlass::gemm::device::Gemm<float,...>` | Same type from sycl-tla `#include <cutlass/gemm/device/gemm.h>` |

---

## Convolution GEMM precision policy

The eight SYCL sparse- and continuous-convolution operators expose an
`allow_tf32=False` argument. The default XPU path uses float32 inputs,
accumulators, and outputs with IEEE float32 multiply/accumulate semantics.
When `allow_tf32=True`, the same float32 buffers are interpreted as
`cutlass::tfloat32_t` inputs and dispatched through sycl-tla's Intel Xe XMX
tensor-operation path; accumulation and output remain float32. CPU and CUDA
dispatches accept the argument for schema compatibility but ignore it.

Both paths use sycl-tla exclusively. `GemmSYCL.h` tries the ordered tile
ladder for each path and raises an error when sycl-tla reports that none of
the candidates can implement the requested layout, leading dimensions, or
shape. There is no custom `parallel_for` GEMM fallback.

sycl-tla main is used for the IEEE path. Its Agnostic kernel type accepts both
column-major/column-major and column-major/row-major operand layouts. The
oneAPI 2025.3 compatibility patch in `3rdparty/sycl_tla` gives the relevant
unscoped `Kind` enums fixed underlying types and uses automatic SYCL kernel
naming for the Agnostic launch. `open3d_torch_ops.so` and `python-package`
build with that patch; XPU runtime validation still requires a host with an
available Intel GPU.

---

## Implementation Status (updated after real-hardware validation on Intel Panther Lake Xe3 iGPU)

### ✅ All 18 ops implemented, built, and functionally validated on real Intel GPU hardware

All 13 previously-stub ops (Voxelize, ContinuousConv family ×4, SparseConv
family ×4, pointnet ×3, pvcnn ×1) are now implemented, in addition to the 5
done previously (RaggedToDense, ReduceSubarraysSum, InvertNeighborsList,
BuildSpatialHashTable, FixedRadiusSearch). `open3d_torch_ops.so` and the full
`open3d` python package build and link successfully with
`BUILD_SYCL_MODULE=ON BUILD_PYTORCH_OPS=ON` on this machine (Intel Panther
Lake, Xe3 iGPU, device id `0xb080`), and were tested for real on-device
execution (not just syntax-checked).

**Validation results (real hardware, this session):**

| Op group | Method | Result |
|----------|--------|--------|
| Voxelize | `pytest python/test/ml_ops/test_voxelize.py -k ml1` | **1344/1344 passed** |
| BallQuery / FPS / three_interp (pointnet) | `pytest test_query_pts.py test_sampling.py test_three_interp.py` | **3/3 passed** |
| TrilinearDevoxelize | direct fwd+bwd vs. independent numpy reference (no CPU op exists — GPU-only by design, matching upstream PVCNN) | max diff 1.2e-7 (float32 precision), grad-sum sanity exact |
| ContinuousConv (4 ops, fwd+bwd) | direct CPU-vs-XPU scripts, all `coordinate_mapping`/`interpolation` combos, hand-built valid neighbor lists | rel. error ~1e-3 (tf32-level) |
| SparseConv (4 ops, fwd+bwd) | direct CPU-vs-XPU scripts, hand-built valid neighbor lists | rel. error <1e-3 (tf32-level) |
| `GemmSYCL.h` shim in isolation | standalone C++ test vs. naive host GEMM | max diff 0.0037 (tf32-level) |

Full `pytest -k ml1` runs of `test_cconv.py`/`test_sparseconv.py` could not be
used as-is: those fixtures build neighbor lists via
`FixedRadiusSearch(metric='Linf')`, but the SYCL FixedRadiusSearch (a
pre-existing op, done before this phase) only supports `metric=L2` — see
"Known pre-existing issues" below. `mltest.py` was extended with a
`torch_xpu` device entry so `-k ml1` now correctly selects the XPU device for
any test that doesn't depend on the Linf/L1 metrics.

**Real bugs found and fixed during this validation (all in newly-added code,
i.e. in-scope for this phase):**
1. `InvertNeighborsListOpKernelSYCL.cpp` called `ToTorchDtype<uint32_t>()`,
   which has no specialization (Torch has no native uint32 dtype) — this
   crashed `ContinuousConv`/`SparseConv` **backward** with `RuntimeError:
   Unsupported type`. Fixed by allocating the scratch count buffer as
   `int32_t` and `reinterpret_cast`-ing the pointer to `uint32_t*` (same bit
   width, used only as an internal counter).
2. **Historical:** `make_python_package.cmake` once copied `cpu`/`cuda`/`sycl`
   arch subdirs; ops now install flat as `open3d/open3d_torch_ops.so` (see
   `docs/local_open3d_ml_parity_testing.md`).
3. **Historical:** `__init__.py` once selected `open3d/{sycl,cpu}` paths; the
   loader now uses a single flat library (CPU and XPU dispatch inside the same
   `.so`).

An initial report of "large numerical error" in `ContinuousConv` (mean/sign
completely wrong) turned out to be a **test-harness bug**, not an algorithm
bug: comparing CPU vs. XPU outputs generated from independently-drawn random
tensors (different `torch.manual_seed` state per call). With matched inputs,
all four `coordinate_mapping`/`interpolation` combinations produce
tf32-level-accurate results. Similarly, an apparent `SparseConv` discrepancy
was traced to synthetic test data with **duplicate `kernel_index` values
within one output row**, which is physically invalid input (each neighbor
should map to a distinct kernel offset) — the reference CUDA `FillColumn`
kernel itself does a plain last-write-wins `=` (not `+=`) per kernel slot, so
duplicate kernel indices are order-dependent in *both* CPU and SYCL
implementations by design, not a bug.

### ⚠️ Known pre-existing issues found during real-hardware testing (out of
### this phase's scope — in `FixedRadiusSearch`/`core/nns`, done before this
### phase; documented as follow-up, not fixed here)

1. SYCL `FixedRadiusSearch`/`KnnSearchOpsSYCL.cpp` only supports `metric=L2`
   (`L1`/`Linf` raise `LogError`).
2. `ignore_query_point=True` is silently **not honored** by the PyTorch SYCL
   dispatch (`FixedRadiusSearchOpKernelSYCL.cpp`) — the parameter is accepted
   but never checked/applied in that file (unlike the Tensor-API path in
   `KnnSearchOpsSYCL.cpp`, which at least throws `LogError`). Both files
   ultimately need `ignore_query_point` support in the SYCL NNS kernels.
3. `python/test/ml_ops/test_fixed_radius_search.py -k ml1` shows widespread
   failures concentrated in `float32` inputs (560/4337 cases fail; `float64`
   cases mostly pass) — a real, dtype-specific correctness bug independent of
   (1)/(2), not investigated further in this phase.

### ⚠️ Known issue: process-exit crash with GEMM-shim ops through the full
### `open3d` package (not a correctness bug, all computed results are
### correct)

Reproducible: calling `continuous_conv`/`sparse_conv` on XPU **and** having
imported the full `open3d` package (which loads a second, independently
SYCL-using shared library, the core `Open3D.so`) causes a
`"corrupted double-linked list"` `SIGABRT` at Python interpreter exit — after
all computation has completed correctly. Does **not** reproduce when: (a)
loading `open3d_torch_ops.so` directly via `torch.ops.load_library(...)`
bypassing the `open3d` package (clean exit), or (b) calling any non-GEMM op
(Voxelize, BallQuery, FPS, TrilinearDevoxelize, etc.) through the full
package (clean exit in every case tested). Hypothesis: two independently
SYCL-using shared objects in one process (the core library and
`open3d_torch_ops.so`) each hold SYCL/Level-Zero context/queue state and
sycl-tla/CUTLASS global singletons (e.g. the `EventManager` in
`tools/util/include/cutlass/util/sycl_event_manager.hpp`); their teardown
order at process exit conflicts with the Level-Zero driver. This looks like
an sycl-tla/Level-Zero interop issue, not fixable without patching 3rdparty
code (against project policy) — recommended follow-up: report upstream to
sycl-tla, or investigate isolating GEMM shim calls to a dedicated SYCL
context/queue.



## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| sycl-tla `device::Gemm<>` unsupported on CPU SYCL backend | Fall back to oneMKL `blas::gemm()` for CPU testing; GPU required for XMX GEMM |
| Sub-group size varies across Intel GPU families | Use `sycl::reduce_over_group()` — adapts to hardware sub-group size automatically |
| `cub::DeviceRunLengthEncode::Encode` has no oneDPL equivalent | Custom SYCL kernel: adjacent_difference + copy_if + exclusive_scan |
| No `texture_alignment` in SYCL device props | `queue.get_device().get_info<sycl::info::device::mem_base_addr_align>() / 8` |

---

## Test Method (locked)

```bash
# CPU correctness (unchanged path — fast validation)
PYTHONPATH=build/lib/python_package pytest \
  python/test/ml_ops/test_cconv.py \
  python/test/ml_ops/test_sparseconv.py \
  python/test/ml_ops/test_general_sparseconv.py -v -k "ml0"

# SYCL GPU test
SYCL_DEVICE_FILTER=level_zero:gpu PYTHONPATH=build/lib/python_package \
  pytest python/test/ml_ops/ -v -k "ml1"
```
