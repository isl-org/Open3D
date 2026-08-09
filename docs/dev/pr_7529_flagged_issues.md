# PR #7529 (`ss/sycl-mlops`) — flagged issues from real-XPU verification

Notes from building and running the full test suite against real Intel Arc A770 XPU
hardware (see `cuddly-swimming-manatee` plan, task "Build + run full test suite on
real XPU hardware"). Everything below was found during that pass; items marked
**Fixed** are already corrected in this branch, the rest are flagged for follow-up.

## Fixed this session

### sycl-tla GEMM epilogue silently produced all-zero output (root cause of most
### ml_ops SYCL failures)

`cutlass::arch::global_load`/`global_store` (`cutlass/arch/memory.h`, vendored via
`3rdparty/sycl_tla`) implemented the IEEE-fp32 (`allow_tf32=False`) epilogue's
final read/write only via CUDA PTX, guarded by
`#if defined(__CUDA_ARCH__) || defined(__SYCL_CUDA_ARCH__)`. `__SYCL_CUDA_ARCH__`
is never defined for genuine SPIR-V/Intel targets, so the `#else` branch
(`CUTE_INVALID_CONTROL_PATH`, a no-op under `-DNDEBUG`) ran instead — the GEMM
epilogue never wrote its result. Fixed by adding a plain predicated
pointer-dereference `#elif defined(__SYCL_DEVICE_ONLY__)` branch, delivered as a
new hunk in `3rdparty/sycl_tla/0001-fix-oneapi-2025.3-ieee-gemm.patch`. Validated
via a standalone repro (26/26 shapes) and via `test_sparseconv.py`,
`test_knn_search.py`, `test_nms.py` (all previously failing, now passing).

### `test_sparseconv_allow_tf32` compared two differently-initialized layers

The test built a fresh, randomly-initialized `SparseConv` layer per call to
`run(allow_tf32)`, so `run(True)` and `run(False)` were never comparing the same
filter weights. Fixed in `python/test/ml_ops/test_sparseconv.py` by constructing
the layer once and toggling `conv.allow_tf32` between calls.

### CUDA multi-batch `KnnSearch` lost its global-index offset (regression, not
### pre-existing)

While re-reading `cpp/open3d/core/nns/KnnSearchOps.cu` for this write-up
(diffing against `main`), found that the combine step at the end of
`KnnSearchCUDA` had been changed to copy `a.IndicesPtr()` (indices local to each
per-batch `points_i` slice) directly into the output, instead of
`a.NeighborsIndex().Add(offset)` (globally-offset indices), which is what `main`
does. This silently broke `neighbors_index` for any CUDA `KnnSearch` call with
`batch_size > 1` — every batch after the first would return point indices local
to its own slice instead of indices into the full concatenated `points` tensor.
This was introduced somewhere in this branch's own history (not present on
`main`), most likely as accidental collateral damage while refactoring the same
function for the `ready_event`/`user_stream` event-bridging change (§2.2 in the
governing plan). **Fixed** by restoring the `.Add(offset)` step.

No CUDA hardware is available in this environment, so this fix is verified by
code inspection only (matches `main`'s known-correct behavior) — flagging so it
gets exercised by CUDA CI/hardware before merge.

## Known failures — pre-existing / out of scope, not fixed

These reproduce identically whether or not this branch's changes are present,
or are explicitly out of scope per prior direction. Listed so they aren't
mistaken for new regressions in a future run.

- **`test_ragged_tensor.py::test_binary_ew_ops[ml1-float64]`** — `RuntimeError:
  level_zero backend failed ... UR_RESULT_ERROR_INVALID_ARGUMENT`. PyTorch-XPU
  does not support float64 on this A770 device. Excluded per explicit user
  direction; not investigated further.
- **`test_linalg.py::test_lu[dtype2-device0]` and `[dtype2-device1]`**
  (float32) — `getrf failed: singular condition detected` on *both* the CPU
  LAPACK backend and the SYCL/oneMKL backend, for the same near-singular 3x4
  test matrix. `test_linalg.py` and `LinalgUtils.h` are byte-identical to
  `main` (`git diff main` is empty), and the failure is backend-independent, so
  this is a pre-existing float32-precision issue with the test's fixture
  matrix, not something introduced by this PR.
- **`test_legacy_headless_rendering.py::test_legacy_visualizer_headless_capture`**
  — `GLFW Error: Failed to detect any supported platform` /
  `EGLOffscreenContext: eglBindAPI(EGL_OPENGL_API) failed`. Environment has no
  display/EGL platform available in this container; unrelated to SYCL/ml_ops.
- **`python-package` CMake target / `pybind11_stubgen`** — fails with
  `Can't find/import '_abc._abc_data'` while generating type stubs for
  `open3d._ml3d.datasets.*`. Pre-existing environment issue (unrelated to any
  code change this session); the actual `.so` artifacts needed for testing are
  installed before this step runs, so it doesn't block test execution, only
  the overall `make python-package` exit code.

## Verified clean

- `python/test/ml_ops/` — every file passes except the float64 case above (run
  per-file in isolation; a single combined `pytest` session across all of
  `ml_ops/` hangs partway through for unrelated reasons — use
  `run_ml_ops_isolated.sh` instead of a single `pytest python/test/ml_ops/`
  invocation).
- `python/test/core/`, `data/`, `geometry/`, `io/`, `ml/`, and top-level tests
  — all pass except `test_lu[dtype2-*]` above.
- `python/test/visualization/` — all pass except the headless-rendering case
  above.

