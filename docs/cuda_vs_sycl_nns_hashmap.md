# CUDA vs. SYCL: nearest-neighbor search and hash map implementations

This is a design note comparing Open3D's CUDA and SYCL backends for the
`core::nns` (KNN / fixed-radius / hybrid) search algorithms and the
`core::HashMap` device backends. It reflects the code after the SYCL
uniform-grid port for fixed-radius/hybrid search (see
`docs/sycl_fr_hybrid_grid_results.md`), which brought the SYCL fixed-radius
and hybrid algorithms to structural parity with CUDA.

## 1. K-Nearest-Neighbor (KNN) search

Both backends share the same high-level strategy — a small-dim/small-k
brute-force path plus a GEMM-based path for the general case — but differ in
the top-k selection machinery.

| Aspect | CUDA (`KnnSearchOps.cu`) | SYCL (`KnnSearchOpsSYCL.cpp`) |
|---|---|---|
| Path selection | 2 paths: brute vs. optimized | 3 paths: Direct, AddMM-fused, AddMM-large-k |
| Brute-force trigger | `dim < 8 && k <= 32` | `dim in [1,8] && k <= 32` (`UseKnnDirect`) |
| Brute-force kernel | 1 thread/query, in-register max-heap (`KnnQueryKernel`) | 1 sub-group/query, SLM point tiles, per-lane sorted top-k + shuffle-merge (`DispatchKnnDirect`) |
| GEMM path | tiled `AddMM(-2·QPᵗ) + ‖p‖²` | same tiled `AddMM(-2·QPᵗ) + ‖p‖²` |
| Top-k on GEMM tiles | FAISS-derived warp/block select (`runL2SelectMin`, `runBlockSelectPair`) | custom heap: GRF-resident (k≤32), scratch-resident (k≤512) |
| Very large k | multi-pass masking (`k > GPU_MAX_SELECTION_K` = 1024/2048) | oneDPL `partial_sort` per query (k > 512) |
| Float stability | uses norms directly | centers points+queries to avoid fp32 cancellation |
| AddMM backend | cuBLAS GEMM | oneMKL GEMM |

**Key divergence:** CUDA reuses FAISS's warp-shuffle `BlockSelect`/`L2Select`
kernels; SYCL hand-rolls a heap-based selector tiered by k, falling back to
oneDPL's serial `partial_sort`. SYCL additionally centers data for numerical
stability, which CUDA does not.

## 2. Fixed-radius and hybrid search

This used to be the biggest structural gap between backends (SYCL used a
brute-force `AddMM` tiling that ignored the hash-table tensors entirely).
That gap has been closed: SYCL now ports CUDA's uniform-grid (cell-list)
algorithm directly, reusing the shared device-agnostic `SpatialHash` /
`ComputeVoxelIndex` helpers in `NeighborSearchCommon.h`. The two
implementations are now algorithmically identical; they differ only in
which vendor primitive plays each role.

| Aspect | CUDA (`FixedRadiusSearchImpl.cuh`) | SYCL (`FixedRadiusSearchSYCLImpl.h`) |
|---|---|---|
| Algorithm | Uniform voxel spatial-hash grid (`BuildSpatialHashTableCUDA`) | Same: uniform voxel spatial-hash grid (`BuildSpatialHashTableSYCL`) |
| Complexity | ~O(N) build, O(1) amortized per query (8 fixed bins) | same |
| Index build | count → `cub::DeviceScan::InclusiveSum` → scatter | count → oneDPL `inclusive_scan` → scatter |
| Query | visits 8 corner-adjacent bins (cell size = 2·radius) | same: visits 8 corner-adjacent bins |
| Fixed-radius | two-pass: `CountNeighborsKernel` then `WriteNeighborsIndicesAndDistancesKernel` | two-pass: `CountNeighborsSYCL` then `WriteNeighborsSYCL` |
| Hybrid | single grid pass, running top-`max_knn` + count, then per-query bubble sort | same: single grid pass in `WriteNeighborsHybridSYCL`, identical running top-k logic and bubble sort (ported line-for-line) |
| Result sort (`sort=true`) | `cub::DeviceSegmentedRadixSort::SortPairs` | oneDPL `sort_by_key`: packed `uint64` radix key `(query_id << 32 \| bit_cast<uint32>(dist))` for `float` (stays on the fast radix path); struct key `{query_id, dist}` + device comparator for `double` (needs all 64 bits for the distance, so no room to pack the segment id) |
| SYCL CPU support | n/a (CUDA-only) | fully supported — the grid algorithm has no GPU-specific sizing dependency, unlike the old AddMM path |
| Parallelism granularity | 1 CUDA thread / query | 1 SYCL work item / query (plain `parallel_for`, no SLM tiling — there is no `subgroups_per_wg`/`tile_points`-style tunable here, unlike the KNN direct kernel) |

**Key convergence:** both backends now build and query the same dense
spatial-hash CSR grid with the same 8-bin visitation rule and the same
hybrid running-top-k + bubble-sort logic; only the underlying
scan/sort/atomic primitives differ (CUB vs. oneDPL/`sycl::atomic_ref`).

## 3. Hash map backends

Both plug into the same `DeviceHashBackend` virtual interface and the same
`core::HashMap`/`HashSet` tensor API, and both are unique-key maps
(`Key -> buf_index_t`, with payload stored separately in
`HashBackendBuffer`) — not multimaps.

| Aspect | CUDA | SYCL |
|---|---|---|
| Backends | `StdGPUHashBackend` (default) + `SlabHashBackend` | single `SYCLHashBackend` |
| Data structure | StdGPU: `stdgpu` open-addressing lib; Slab: warp-slab bucket chaining | hand-written open addressing, packed 64-bit slots + CAS, fingerprint fast-reject |
| Third-party dep | `stdgpu` / `slab-hash` | none (aligns with AGENTS.md "avoid stdgpu") |
| Device-side find | `GetImpl()` → `map.find(key)` in kernel | `GetDeviceLookup()` → `SYCLHashDeviceLookup::Find(key)` (plain loads, no atomics, for read-only lookups) |
| `GetActiveIndices` | backend-specific (warp-cooperative) | work-group `exclusive_scan_over_group` + one global `fetch_add` per group |
| Keys | `MiniVec<int, dim>`, dim 1–6, Int16/32/64 | identical |
| Buffer accessor | `CUDAHashBackendBufferAccessor` (`DeviceAllocate`/`Free`) | `SYCLHashBackendBufferAccessor` (same API) |

**Shared growth model** (identical on both backends): the backend
`Reserve()` is a no-op; capacity growth is driven by `HashMap::Reserve`
doing a whole-table rebuild (export actives → Free → Allocate →
re-insert). Both size the table for ~0.5 load factor.

**Key divergence:** CUDA leans on external libraries (`stdgpu` default,
`slab-hash` alternative) offering two structurally different maps; SYCL is a
single self-contained open-addressing implementation whose distinctive
feature is a stored fingerprint per slot for fast probe rejection (see
`cpp/open3d/core/hashmap/SYCL/SYCLHashBackend.h` for why this doesn't need a
separate hash function from the main table probe).

## 4. Shared infrastructure and primitive substitutions

The implementations are architecturally parallel, differing mainly in which
vendor primitive fills each role:

| Primitive | CUDA | SYCL |
|---|---|---|
| Dense GEMM (`AddMM`) | cuBLAS | oneMKL |
| Parallel loop | `__global__` kernels | `ParallelFor` / `queue.parallel_for` |
| Atomics | `atomicAdd` | `sycl::atomic_ref` |
| Prefix sum | `cub::DeviceScan::InclusiveSum` | work-group `exclusive_scan_over_group`, oneDPL `inclusive_scan`, or host scan (context-dependent) |
| Sort | CUB radix / segmented radix (`cub::DeviceSegmentedRadixSort`) | oneDPL `sort_by_key` (segmented via composite/packed keys) / `partial_sort` / per-query insertion or bubble sort for small, bounded outputs |
| Geometry helpers | `NeighborSearchCommon.h` (`HOST_DEVICE`) | same file, reused unchanged |

Shared abstraction layers vs. the vendor-specific layer beneath them:

```
                    core::HashMap / NNSIndex / DeviceHashBackend
                    NeighborSearchCommon.h: SpatialHash, ComputeVoxelIndex
                                    │
              ┌─────────────────────┴─────────────────────┐
              │ CUDA                                       │ SYCL
   KNN:        brute-heap + cuBLAS/FAISS select              Direct (SLM) / AddMM-heap / oneDPL
   FixedRadius/Hybrid: voxel grid + CUB scan/sort             voxel grid + oneDPL scan/sort (ported)
   Hash:       StdGPU / Slab (third-party)                   custom open addressing (dependency-free)
```

## Bottom line

- **KNN:** functionally equivalent designs (brute-force + GEMM paths),
  different top-k selectors (FAISS warp-select vs. custom heap/oneDPL);
  SYCL additionally centers data for float32 stability.
- **Fixed-radius/Hybrid:** now structurally identical — both are the same
  uniform spatial-hash grid algorithm, differing only in the underlying
  scan/sort/atomic primitives (CUB vs. oneDPL). SYCL also runs this on CPU
  devices, which the old AddMM path could not.
- **Hash maps:** same interface and same unique-key + whole-table-rehash
  model; CUDA uses third-party libraries (two backends), SYCL is a single
  dependency-free open-addressing map with slot fingerprints.

The recurring pattern is that Open3D keeps a device-agnostic API and
geometry layer, then swaps vendor primitives (cuBLAS↔oneMKL,
CUB↔oneDPL/work-group scans, stdgpu/slab↔hand-written) underneath. Following
the fixed-radius/hybrid grid port, the *algorithms themselves* are now
aligned across all three `core::nns` search modes; the remaining
differences are all at the primitive-substitution level described in
section 4.
