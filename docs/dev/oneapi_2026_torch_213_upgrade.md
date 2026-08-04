# oneAPI 2026.0 + PyTorch 2.13 (XPU) upgrade

## Goal

Determine whether the `free(): double linked list corruption` crash seen at
interpreter shutdown in `pytest python/test/ml_ops/test_cconv.py` (SYCL device)
is resolved by moving from PyTorch 2.10+xpu / oneAPI 2025.3 to
PyTorch 2.13+xpu / oneAPI 2026.0.

## Requirements

1. `python-package` target builds with oneAPI 2026.0 (`icpx`) and links against
   PyTorch 2.13.0+xpu.
2. All other MKL usage (CPU BLAS/LAPACK) keeps working unchanged; no new runtime
   dependencies that are absent from the `onemkl-sycl-*` pip packages that a
   torch-xpu environment already ships.
3. Windows and non-SYCL oneAPI code paths untouched.

## Test method

```bash
cd build
source /opt/intel/oneapi/2026.0/oneapi-vars.sh
source ~/Documents/o3d.venv/bin/activate
cmake --build . --parallel $(nproc) --target python-package
export PYTHONPATH=$PWD/lib/python_package
pytest ../python/test/ml_ops/test_cconv.py
```

Pass criteria: tests pass **and** the process exits cleanly (no
`free(): double linked list corruption`, exit code 0).

## Findings

### MKL SYCL static libraries removed in oneAPI 2026.0

| | 2025.3 | 2026.0 |
|---|---|---|
| `libmkl_sycl.a` | present | **removed** |
| `libmkl_sycl*.so` | present | present |

`libmkl_sycl.so` is a GNU ld linker script:

```
INPUT(-lmkl_sycl_blas -lmkl_sycl_lapack -lmkl_sycl_sparse -lmkl_sycl_dft
      -lmkl_sycl_vm -lmkl_sycl_rng -lmkl_sycl_stats -lmkl_sycl_data_fitting)
```

Linking the umbrella would add `DT_NEEDED` entries for the `vm`, `stats` and
`data_fitting` domains, which are **not** shipped by the oneMKL pip packages
(`onemkl-sycl-{blas,dft,lapack,rng,sparse}` only). Open3D only uses the `blas`
domain (`Matmul`, `AddMM`) and the `lapack` domain (`LU`, `SVD`, `Solve`,
`Inverse`, `LeastSquares`), so only those two are linked.

`libmkl_sycl_blas.so.6` / `libmkl_sycl_lapack.so.6` depend only on `libsycl.so.9`
and `libdl.so.2` — not on `libmkl_core.so`. Mixing them with the *static* CPU MKL
(`mkl_intel_ilp64`, `mkl_tbb_thread`, `mkl_core`) therefore does not create two
copies of the MKL CPU runtime in the process.

## Implementation steps

- [x] Confirm the static/shared MKL SYCL library layout change between 2025.3 and 2026.0.
- [x] Confirm which oneMKL domains Open3D actually calls (blas, lapack only).
- [x] `3rdparty/find_dependencies.cmake`: already links `mkl_sycl_blas` /
      `mkl_sycl_lapack` as shared libraries on Linux, keeping the CPU MKL
      static (pre-existing `if(BUILD_SYCL_MODULE AND NOT WIN32)` branch).
- [x] CI/build-system version bump: `util/ci_utils.sh` (`TORCH_VER=2.13`),
      `python/requirements_sycl.txt` (`dpcpp-cpp-rt==2026.0.0`),
      `docker/docker_build.sh` and `util/install_oneapi_windows.ps1` (moved
      from deprecated `intel/cpp-essentials` / `intel-oneapi-base-toolkit` to
      the unified `intel/oneapi-toolkit:2026.0.1-devel-ubuntu22.04` image and
      the `intel-oneapi-toolkit-2026.0.0.193_offline.exe` Windows installer).
- [ ] Rebuild `python-package` with oneAPI 2026.0 and torch 2.13.
- [ ] Run `pytest python/test/ml_ops/test_cconv.py`, and the rest of
      `python/test/ml_ops/` as a regression check.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `libmkl_sycl_blas.so.6` not found at runtime in a wheel install | The `onemkl-sycl-blas`/`onemkl-sycl-lapack` pip packages are already pulled in by `torch==*+xpu`; oneAPI's `oneapi-vars.sh` covers local dev. Wheel `RPATH`/dependency declaration is a follow-up for release builds. |
| Two MKL runtimes in one process (static CPU + shared SYCL) | Verified `libmkl_sycl_{blas,lapack}.so.6` have no `libmkl_core.so` `DT_NEEDED`. |
| oneAPI 2025.3 builds regress | `mkl_sycl_blas`/`mkl_sycl_lapack` shared libraries exist in 2025.3 too, so the same link line works for both. |

## Status log

- Baseline: torch 2.10+xpu / oneAPI 2025.3 — `test_cconv.py` tests pass but the
  process aborts at shutdown with `free(): double linked list corruption`.
- Environment now upgraded to torch 2.13.0+xpu and oneAPI 2026.0 (mkl 2026.0
  pip packages).
- Result of the rebuild + test run: TBD.

## Follow-ups

- Declare/bundle the oneMKL SYCL shared libraries for released SYCL wheels so
  `libmkl_sycl_{blas,lapack}.so.6` resolve without `oneapi-vars.sh`.
