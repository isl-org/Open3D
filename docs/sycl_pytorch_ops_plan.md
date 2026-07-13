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

## Implementation Status

### ✅ Done

| Commit | What |
|--------|------|
| `84a11feee` | CUTLASS v1→v4.2.1 migration; sycl-tla CMake scaffold; 8 `.cuh` files migrated |
| `939d0f374` | CMake scaffolding (`if(BUILD_SYCL_MODULE)` block, all 20 stub sources); 5 misc op impl headers + PyTorch kernels; refactoring to eliminate duplication |

**Implemented op kernels (5 of 18):**

| Op | Impl header | PyTorch kernel | Notes |
|----|-------------|----------------|-------|
| RaggedToDense | `ml/impl/misc/RaggedToDenseSYCL.h` | `misc/RaggedToDenseOpKernelSYCL.cpp` | 1-D parallel_for |
| ReduceSubarraysSum | `ml/impl/misc/ReduceSubarraysSumSYCL.h` | `misc/ReduceSubarraysSumOpKernelSYCL.cpp` | 1-D parallel_for |
| InvertNeighborsList | `ml/impl/misc/InvertNeighborsListSYCL.h` | `misc/InvertNeighborsListOpKernelSYCL.cpp` | atomic histogram + oneDPL scan |
| BuildSpatialHashTable | *(delegates to `core/nns/kernel/FixedRadiusSearchSYCLImpl.h`'s `BuildSpatialHashTableSYCLRaw<T>()`)* | `misc/BuildSpatialHashTableOpKernelSYCL.cpp` | 54-line thin wrapper |
| FixedRadiusSearch | *(delegates to `core/nns/kernel/FixedRadiusSearchSYCLImpl.h`'s `CountNeighborsSYCL`/`WriteNeighborsSYCL`)* | `misc/FixedRadiusSearchOpKernelSYCL.cpp` | reuses NNS impl |

**Refactoring done:**
- `BuildSpatialHashTableSYCLRaw<T>()` extracted into `FixedRadiusSearchSYCLImpl.h` — shared by Open3D Tensor API and PyTorch dispatch
- `CountNeighborsSYCL` renamed to `CountIndexOccurrencesSYCL` in `InvertNeighborsListSYCL.h` to avoid name collision

---

### 🔴 Stub files only — not yet implemented (13 of 18)

| Group | Files | Key challenge |
|-------|-------|---------------|
| **Voxelize** | `VoxelizeSYCL.h`, `VoxelizeOpKernelSYCL.cpp` | 9 kernels; oneDPL sort_by_key; custom RLE kernel replacing `cub::DeviceRunLengthEncode` |
| **SparseConv** (4 ops) | `SparseConvSYCLKernels.{h,cpp}`, 4 `*SYCL.h`, 4 `*OpKernelSYCL.cpp` | sycl-tla `device::Gemm<>`; FillColumn im2col with sub-group reduce |
| **ContinuousConv** (4 ops) | `ContinuousConvSYCLKernels.{h,cpp}`, 4 `*SYCL.h`, 4 `*OpKernelSYCL.cpp` | sycl-tla GEMM + trilinear interp; warp-reduce normalizer → `sycl::reduce_over_group` |
| **Pointnet** (3 ops) | `BallQueryKernelSYCL.cpp`, `InterpolateKernelSYCL.cpp`, `SamplingKernelSYCL.cpp` | FPS: shared-memory warp reduce → `sycl::nd_range` + `sycl::local_accessor` tree reduce |
| **pvcnn** (1 op) | `TrilinearDevoxelizeKernelSYCL.cpp` | `nd_range` + `local_accessor` for scatter |

**Impl headers still needed:**
```
ml/impl/continuous_conv/ContinuousConvSYCLKernels.h + .cpp
ml/impl/continuous_conv/ContinuousConv{,Backprop,Transpose,TransposeBackprop}SYCL.h
ml/impl/sparse_conv/SparseConvSYCLKernels.h + .cpp
ml/impl/sparse_conv/SparseConv{,Backprop,Transpose,TransposeBackprop}SYCL.h
ml/impl/misc/VoxelizeSYCL.h
ml/impl/contrib/BallQuerySYCL.h
ml/impl/contrib/InterpolatePointsSYCL.h
ml/impl/contrib/TrilinearDevoxelizeSYCL.h
```

---

### 🔴 Build — Blocked (as of 2026-07-13)

cmake configure fails on this machine (no Intel GPU) at `FindSYCLToolkit.cmake` inside PyTorch's cmake:

```
CMake Error: list GET given empty list
  at FindSYCLToolkit.cmake:61 (parse_sycl_compiler_version)
```

**Root cause:** PyTorch's bundled `FindSYCLToolkit.cmake` parses `icx --version` output but
fails when env vars are not set via `setvars.sh`. The workaround is:

```bash
source /opt/intel/oneapi/setvars.sh  # sets CMPLR_ROOT, MKL_DIR, TBB_DIR automatically
cmake -S . -B /path/to/build \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DBUILD_SYCL_MODULE=ON -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_CUDA_MODULE=OFF \
  -DGLIBCXX_USE_CXX11_ABI=ON \
  -DPython3_EXECUTABLE=/mnt/seagate_4tb_b/ssheorey/venv-o3d-xpu/bin/python
```

**Decision:** Transfer branch to Intel GPU machine for build + test.

---

## Next Steps (on Intel GPU machine)

1. **Clone / pull** `ss/sycl-mlops` branch
2. **Install PyTorch XPU:** `pip install torch==2.10.0+xpu --index-url https://download.pytorch.org/whl/xpu`
3. **Configure:**
   ```bash
   source /opt/intel/oneapi/setvars.sh
   cmake -S /path/to/Open3D -B /path/to/build \
     -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
     -DBUILD_SYCL_MODULE=ON -DBUILD_PYTORCH_OPS=ON \
     -DBUILD_CUDA_MODULE=OFF -DBUILD_UNIT_TESTS=OFF \
     -DGLIBCXX_USE_CXX11_ABI=ON \
     -DPython3_EXECUTABLE=$(which python)
   ```
4. **Build torch ops target:** `cmake --build build --parallel $(nproc) --target open3d_torch_ops`
5. **Fix compile errors** in stub files iteratively
6. **Implement remaining 13 op kernels** (see stubs)
7. **Build python-package:** `cmake --build build --parallel $(nproc) --target python-package`
8. **Test:**
   ```bash
   SYCL_DEVICE_FILTER=level_zero:gpu PYTHONPATH=build/lib/python_package \
     pytest python/test/ml_ops/ -v -k "ml1"
   ```
9. **Style:** `python util/check_style.py --apply`
10. **Commit + PR**

---

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
