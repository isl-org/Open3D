# SYCL ML ops: order-changing follow-ups (not yet implemented)

This is a punch list of SYCL ML op optimizations that would give real speedups but
change the *ordering* of results (which neighbor is picked among ties, which of several
valid points is retained, etc.). They were intentionally **not** implemented as part of
the initial launch-config/reduction sweep because several correctness tests use
`np.testing.assert_equal` (exact match) and depend on the current, ordering-sensitive
serial-scan implementation. Each item below needs its test updated to an
order-insensitive comparison (or a documented tie-break rule matching the new
implementation) before the kernel is rewritten.

See `cpp/open3d/ml/impl/GemmSYCL.h`, `cpp/open3d/ml/impl/misc/ReduceSubarraysSumSYCL.h`,
and the `nd_range`-launch conversions across `cpp/open3d/ml/contrib/` and
`cpp/open3d/ml/impl/misc/` for the optimizations that *were* applied in this pass
(order-preserving launch-config and coalescing changes only).

## 1. BallQuery

- **File:** `cpp/open3d/ml/contrib/BallQuerySYCL.h` (`BallQuerySYCL`).
- **Current implementation:** one work-item per (batch, query point); each work-item
  serially scans all `n` candidate points in index order and records up to `nsample`
  points whose distance is within `radius`, in scan order.
- **Reusable infra to call instead:** the spatial-hash grid in
  `cpp/open3d/core/nns/kernel/FixedRadiusSearchSYCLImpl.h` —
  `BuildSpatialHashTableSYCL`/`BuildSpatialHashTableSYCLRaw` to bucket points into a
  uniform grid (cell size `2*radius`), then `WriteNeighborsHybridSYCL` (capped
  top-`max_knn`-in-radius, single pass) to gather up to `nsample` neighbors per query.
  This turns an O(n) scan per query into an O(cell occupancy) scan.
- **Ordering impact:** the hash grid visits candidate points in bucket/cell order, not
  by their original index. The current CUDA/SYCL port scans in ascending point index and
  keeps the *first* `nsample` matches in that order; a grid-based rewrite would pick a
  different (but still valid, within-radius) subset/order whenever more than `nsample`
  points qualify.
- **Test to relax:** `python/test/ml_ops/test_query_pts.py` (`test_query_pts`) uses
  `np.testing.assert_equal(ans, expected)` against a fixed reference `.npy` file. This
  would need either a regenerated reference matching the new traversal order, or a
  relaxed comparison (e.g. compare the *set* of neighbor indices per query, or the
  gathered point coordinates rather than raw indices) if `nsample` can be exceeded by the
  true neighbor count.
- **Expected speedup class:** large for scenes with many candidate points per batch
  (currently O(n) per query dominates for big point clouds); the hash grid is
  near-constant time per query for a roughly uniform point density.

## 2. RoiPool point-selection kernel

- **File:** `cpp/open3d/ml/contrib/RoiPoolKernelSYCL.cpp` (kernel 2, the per-(batch,box)
  point-collection loop).
- **Current implementation:** one work-item per (batch, box); each work-item serially
  scans all `pts_num` points in index order and collects up to `sampled_pts_num` indices
  of points found inside the box, in scan order; if fewer than `sampled_pts_num` points
  are found, it pads by duplicating already-collected indices modulo `cnt`.
- **Reusable infra to call instead:** no existing Open3D primitive directly matches this
  "gather up to N members of a per-group predicate, in a work-group" shape ; the
  most direct route is a **work-group-per-box** kernel where work-items grid-stride over
  `pts_num`, each recording a local match, followed by a work-group-scan/compaction
  (parallel prefix sum over the boolean matches within the box, using the same
  `sycl::exclusive_scan_over_group`/SLM technique as e.g. `FillColumnSYCL`'s
  `reduce_over_group`, generalized to a scan) to compute each matched point's output slot
  in parallel instead of a serial `cnt++`.
- **Ordering impact:** parallelizing the point scan (e.g. splitting `pts_num` across
  work-items in a work-group) means points are no longer necessarily collected in
  strictly ascending index order unless the scan/compaction is explicitly designed to
  preserve it (a stable compaction is possible but adds complexity — an
  `exclusive_scan_over_group` based compaction *does* preserve original index order if
  work-items still process points in index order per sub-range, so this rewrite is
  *likely* order-preserving if implemented carefully; still flagged here because it
  hasn't been done/tested).
- **Test to relax:** `python/test/ml_ops/test_roi_pool.py` uses
  `np.testing.assert_equal` on `pooled_features`/`pooled_empty_flag`. If a stable
  (index-order-preserving) compaction is used, the test may not need changes; if not,
  the test would need to compare point *sets* per box rather than exact per-slot values
  (since which of the >`sampled_pts_num` points are kept, when padding, could differ).
- **Expected speedup class:** moderate — the per-box scan over all points is the
  dominant cost of the 3-kernel RoiPool pipeline for large point counts; parallelizing it
  turns an O(pts_num) serial loop into an O(pts_num / work-group-size) strided loop plus
  a scan.

## 3. three_nn top-3 scan

- **File:** `cpp/open3d/ml/contrib/InterpolatePointsSYCL.h` (`ThreeNNSYCL`).
- **Current implementation:** one work-item per (batch, point); each work-item serially
  scans all `m` "known" points in index order, maintaining a running top-3
  smallest-distance list (insertion-sort style) with first-found tie-breaking (a later
  point with an equal distance to an already-recorded one is not swapped in, since the
  scan uses strict `<` comparisons).
- **Reusable infra to call instead:** `core/nns/kernel/KnnSearchSYCLImpl.h`'s
  small-K brute-force path (`DispatchKnnDirect`, sub-group-per-query with SLM
  double-buffered tiles and `sycl::select_from_group` shuffle-merge for the per-lane
  sorted top-K) is built for exactly this shape (fixed small K, brute-force over all
  candidates) and should be substantially faster than the serial per-point scan.
- **Ordering impact:** `DispatchKnnDirect`'s merge order and tie-breaking (which
  candidate wins when two points are equidistant) is not guaranteed to match the serial
  scan's strict first-found-wins rule.
- **Test to relax:** `python/test/ml_ops/test_three_nn.py` and (transitively, since
  `three_interpolate`'s indices come from `three_nn`) `test_three_interp.py` use
  `np.testing.assert_equal` on the returned indices/distances. Random test data rarely
  produces exact ties, so in practice this is low risk, but the tests should either
  explicitly avoid/document tie scenarios or compare distances with a small tolerance
  and indices only among tied-distance groups.
- **Expected speedup class:** large for bigger `m` (number of known points per batch) —
  same O(m) serial-scan-per-query bottleneck as BallQuery, replaced by a tiled,
  sub-group-cooperative brute-force KNN.

## How to proceed on any of these

1. Get user approval for the specific op and the test change.
2. Update the test to an order-insensitive assertion (or regenerate exact reference data
   matching the new algorithm, if determinism can be preserved).
3. Implement the kernel using the reusable infra named above.
4. Run `pytest python/test/ml_ops/<test>.py` and the relevant
   `../Open3D-ML/tests/test_models.py` case(s) to confirm parity.
