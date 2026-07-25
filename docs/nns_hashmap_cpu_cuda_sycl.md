# CPU, CUDA, and SYCL: `core::nns` and `HashMap` backends

Design note for how Open3D picks and implements nearest-neighbor search and
device hash maps. **CUDA and SYCL** share the same GPU-oriented algorithms
below; **native CPU** (`Device("CPU:0")`) uses different structures. SYCL can
target a CPU device and then follows the SYCL column, not the CPU column.

Deeper detail: `NanoFlannImpl.h`, `KnnSearchOpsSYCL.cpp`, `FixedRadiusSearchSYCLImpl.h`,
`SYCLHashBackend.h`.

## Backend routing

| Tensor device | NNS (typical entry: `NearestNeighborSearch`) | `HashMap` |
|---|---|---|
| **CPU** | `NanoFlannIndex` — nanoflann **KD-tree** (L2); KNN, radius, hybrid, multi-radius on the tree | `TBBHashBackend` (`tbb::concurrent_unordered_map`) |
| **CUDA** | `KnnIndex` + `FixedRadiusIndex` (GEMM KNN; **uniform voxel grid** for radius/hybrid) | `StdGPUHashBackend` (default) or `SlabHashBackend` |
| **SYCL** | Same index classes as CUDA; grid + KNN paths run on SYCL queues (including SYCL CPU) | `SYCLHashBackend` (in-tree open addressing) |

**CPU nuance:** `FixedRadiusIndex` on a CPU tensor can still build the same
voxel spatial-hash grid as CUDA (`BuildSpatialHashTableCPU`). The
`NearestNeighborSearch` facade on CPU does **not** use that path for
`FixedRadiusIndex()` / `HybridIndex()` — it keeps a single **NanoFlann** index
instead. Direct `FixedRadiusIndex` use is for callers that want the grid on CPU.

NNS voxel grids are **not** `core::HashMap`; they are dedicated CSR hash tables
in `FixedRadiusIndex`.

## KNN search

| | CPU | CUDA | SYCL |
|---|---|---|---|
| Index | KD-tree (nanoflann) | `KnnSearchOps.cu` | `KnnSearchOpsSYCL.cpp` |
| Small dim / k | tree search | brute if `dim < 8` and `k ≤ 32` | **Direct** if `dim ≤ 8` and `k ≤ 32` |
| General case | tree search | tiled `AddMM(-2·QPᵗ)` + ‖p‖²; FAISS warp/block top-k | same AddMM; custom heap top-k; **centers** data on AddMM paths |
| Large k | tree | multi-pass mask if `k > GPU_MAX_SELECTION_K` | oneDPL `partial_sort` if `k > 512` |
| GEMM | — | cuBLAS | oneMKL |

Main GPU differences: top-k machinery (FAISS vs heap/oneDPL), SYCL **dim 8**
on the direct path (CUDA uses GEMM at dim 8), optional SYCL `tile_bytes`
(`NeighborSearchCommon.h`).

## Fixed-radius and hybrid search

CUDA and SYCL (and `FixedRadiusIndex` on CPU) use the same **uniform voxel
grid**: cell size `2·radius`, **8 bins** per query, metrics **L1 / L2 / Linf**.
Build = count → prefix sum → scatter; fixed-radius = count pass then write;
hybrid = one pass with running top-`max_knn` + local sort. Vendor swap: **CUB**
(CUDA) vs **oneDPL** (SYCL) for scan and segmented sort.

Via **`NearestNeighborSearch` on CPU**, radius and hybrid search use **NanoFlann**
on the KD-tree, not this grid.

## Hash map (`DeviceHashBackend`)

Same API everywhere: unique keys → `buf_index_t`, values in
`HashBackendBuffer`. Backend `Reserve()` is a no-op; **`HashMap::Reserve`**
rebuilds the whole table (~0.5 load factor). Rehash also when
`GetNonEmptyCount() + batch` would exceed capacity (tombstones on SYCL).

| | CPU (TBB) | CUDA | SYCL |
|---|---|---|---|
| Implementation | TBB concurrent map | stdgpu (default) or warp **slab** | Packed slots, fingerprints, linear probing, **no stdgpu** |
| Device lookup in kernels | n/a | `map.find` / slab | `SYCLHashDeviceLookup` (read-only) |
| Notable SYCL semantics | — | — | Bulk host buffer reserve before insert; **buf_indices** need not be dense — use `GetActiveIndices()` when compact rows matter |

## Primitive substitution (GPU)

Shared geometry: `NeighborSearchCommon.h` (`SpatialHash`, `ComputeVoxelIndex`).
Otherwise: cuBLAS ↔ oneMKL, CUB ↔ oneDPL, stdgpu/slab ↔ hand-written SYCL hash.

**Summary:** Native **CPU** NNS is **NanoFlann** (+ TBB hash maps). **CUDA and
SYCL** align on GEMM KNN and voxel-grid radius/hybrid; they differ in libraries
and a few KNN/hash semantics above. SYCL-on-CPU follows the SYCL column, not
NanoFlann.
