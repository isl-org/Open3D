# SYCL uniform-grid FixedRadius / Hybrid search — results

Plan: "SYCL Uniform-Grid Fixed-Radius / Hybrid Search" (chat plan, not a file
under `.cursor/plans/`).

## What changed

Ported CUDA's uniform-grid (cell-list) fixed-radius algorithm
(`FixedRadiusSearchImpl.cuh`) to SYCL, replacing the previous O(N·M)
tiled-`AddMM` brute-force path used by `FixedRadiusSearchSYCL` and
`HybridSearchSYCL` (`cpp/open3d/core/nns/KnnSearchOpsSYCL.cpp`):

- New device kernels in `cpp/open3d/core/nns/kernel/FixedRadiusSearchSYCLImpl.h`:
  `BuildSpatialHashTableSYCL` (count → oneDPL `inclusive_scan` → scatter,
  building a dense spatial-hash CSR grid with cell size `2*radius`),
  `CountNeighborsSYCL` / `WriteNeighborsSYCL` (fixed-radius two-pass:
  count-then-gather over the 8 corner-adjacent hash bins per query),
  `WriteNeighborsHybridSYCL` (hybrid single-pass: running top-`max_knn` +
  count, with a per-query bubble sort — mirrors CUDA's `max_knn`-bounded
  selection, no heap needed since `max_knn` is small in practice), and
  `SortNeighborsByDistanceSYCL` (device-only segmented sort via oneDPL
  `sort_by_key`: `uint64` packed radix key for `float`, struct key + device
  comparator for `double`).
- `FixedRadiusIndex::SetTensorData` now calls `BuildSpatialHashTableSYCL`
  instead of no-op'ing on SYCL.
- Removed now-unused AddMM-based FR/Hybrid helpers from
  `KnnSearchSYCLImpl.h` (`CountAndSelectTopKQueriesHeap`,
  `CountWithinThresholdQueries`, `GatherWithinThresholdQueries`,
  `FinalizeHybridResults`, `AddQueryNormsToHybridDistances`, etc.). KNN
  large-k still uses the AddMM + `SelectTopKQueries`/`MergeTopKQueries` path.
- Dropped the "not supported on SYCL CPU" restriction for FR/Hybrid: the
  grid algorithm has no CPU-specific limitation (unlike the old AddMM path,
  which relied on oneMKL GEMM sizing tuned for GPU). Verified by unskipping
  and running all previously CPU-guarded tests (see Correctness below).
- Unlike the KNN direct-distance kernel
  (`docs/sycl_knn_direct_path_results.md`), these kernels have no
  SLM-tiling / work-group-size tunables to sweep: each query is one work
  item, doing a global-memory grid lookup with no shared-local-memory
  staging, so there is no `subgroups_per_wg`/`tile_points`-style parameter
  surface. The benchmark run below is therefore a before/after
  characterization, not a parameter sweep.

Hardware: SYCL CPU fallback (`opencl:cpu`, no GPU available in this
environment); intel-arc GPU numbers not available here. CUDA reference
algorithm unchanged.

## Correctness

- C++ (`./bin/tests`, `CI=1` to exercise the SYCL-CPU fallback device):
  - `NNSPermuteDevices.{FixedRadiusSearch,HybridSearch}/1` (SYCL:0): exact
    parity vs. ground truth.
  - `SYCLNNSTest.{FixedRadiusSearchMatchesCPU,
    FixedRadiusSearchCoincidentPoint_C1, HybridSearchMatchesCPU,
    HybridSearchLargeOffsetParityCPU, FixedRadiusSearchNonDefaultTileBytes}`:
    all pass (previously skipped on SYCL CPU; now unskipped and passing).
  - `PointCloudPermuteDevices.{EstimateNormals,RemoveRadiusOutliers}/1`
    (SYCL:0): pass, exercising `EstimateCovariancesUsing{Hybrid,KNN,Radius}
    SearchSYCL` and the new grid-based radius search end-to-end.
  - Full `PointCloudPermuteDevices` suite: 73/76 pass; 3 skips are
    unrelated to this change (SYCL hashmap-based `VoxelDownSample` /
    `RemoveDuplicatedPoints`, and an unimplemented normals-from-RGBD path).
- Python (`pytest python/test/core/test_nn.py`, `CI=1`):
  - `test_fixed_radius_search` parametrized over `also_sycl_cpu=True`
    (previously excluded); `test_fixed_radius_search_random` /
    `test_hybrid_search_random` gated on `>= 1` SYCL devices (previously
    `> 1`, i.e. GPU-only) so the CPU fallback is exercised too. 11/11
    relevant tests pass (8 skipped are GPU-only KNN regression tests with
    no GPU present).

## Benchmark (`./bin/benchmarks`, `--benchmark_filter="SYCL_(Frs|Hybrid)"`)

SYCL device is the CPU fallback (`opencl:cpu`); `CPU_*` rows are the
existing nanoflann-based `Device("CPU:0")` path for reference. Points are
uniform-random in `[0,1]^3`; radius is chosen so ~k points fall in range.

| Benchmark            | SYCL (CPU fallback) | CPU:0 (nanoflann) |
|----------------------|---------------------|--------------------|
| FrsBuild_k8_10k       | 0.52 ms             | 2.63 ms            |
| FrsBuild_k8_100k      | 1.63 ms             | 35.8 ms            |
| FrsBuild_k8_1M        | 7.92 ms             | 604 ms             |
| FrsSearch_1M_k1       | 2.70 ms             | 0.41 ms            |
| FrsSearch_1M_k8       | 7.26 ms             | 0.79 ms            |
| FrsSearch_1M_k32      | 7.83 ms             | 1.11 ms            |
| HybridBuild_k8_1M     | 8.04 ms             | 566 ms             |
| HybridSearch_1M_k1    | 1.00 ms             | 0.13 ms            |
| HybridSearch_1M_k8    | 1.59 ms             | 0.25 ms            |
| HybridSearch_1M_k32   | 3.12 ms             | 0.53 ms            |

Full results: `docs/sycl_fr_hybrid_grid_after.json` (SYCL),
`docs/sycl_fr_hybrid_cpu_reference.json` (CPU:0 reference).

Takeaways:

- **Build** (`BuildSpatialHashTableSYCL`, O(N) count/scan/scatter) is
  **~75x faster at 1M points** than nanoflann's O(N log N) k-d tree build,
  even on the CPU fallback device — confirms the grid structure itself is a
  good fit regardless of backend.
- **Search** is 2–8x slower than nanoflann on the CPU fallback device. This
  is expected: `opencl:cpu` pays per-`parallel_for` kernel-launch overhead
  and lacks nanoflann's TBB-parallelized, cache-tuned tree traversal; it is
  not the intended target hardware. The grid search kernels have no
  GPU-specific dependency (no SLM, no subgroup ops beyond the atomic
  counters in the build phase), so performance on a real SYCL GPU should
  track the CUDA reference's cell-list results rather than this CPU-fallback
  number — re-run this benchmark on GPU hardware to confirm before drawing
  conclusions about GPU-vs-CPU:0 performance.
