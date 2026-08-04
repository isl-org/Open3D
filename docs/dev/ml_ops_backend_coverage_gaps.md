# PyTorch ML ops: CPU/CUDA/SYCL backend coverage gaps

Audit of `cpp/open3d/ml/pytorch/**/*Ops.cpp` dispatch files to find ops that do
not support all three backends (CPU, CUDA, SYCL), why, and whether the gap is
actually exercised by any model in the sibling `Open3D-ML` repo.

## Ops with a backend missing

| Op (file) | CPU | CUDA | SYCL | Root cause |
|---|:-:|:-:|:-:|---|
| `VoxelPooling` ([VoxelPoolingOps.cpp](../../cpp/open3d/ml/pytorch/misc/VoxelPoolingOps.cpp)) | Y | N | N | No CUDA implementation ever existed (`TORCH_CHECK(false, "VoxelPooling does not support CUDA")`, no `#ifdef` guard at all); this PR did not add SYCL either. `cpp/open3d/ml/impl/misc/VoxelPooling.h` only has a CPU implementation. |
| `KnnSearch` (legacy, [KnnSearchOps.cpp](../../cpp/open3d/ml/pytorch/misc/KnnSearchOps.cpp) — distinct from the newer `core::nns`-based op) | Y | N | N | `KnnSearchCPU` is built on `nanoflann` (`core/nns/NanoFlannImpl.h`), a CPU-only header-only KD-tree library with no GPU/SYCL port. |
| `MultiRadiusSearch` (legacy, [RadiusSearchOps.cpp](../../cpp/open3d/ml/pytorch/misc/RadiusSearchOps.cpp)) | Y | N | N | Same root cause: `RadiusSearchCPU` also wraps nanoflann. |
| `TrilinearDevoxelize` ([TrilinearDevoxelizeOps.cpp](../../cpp/open3d/ml/pytorch/pvcnn/TrilinearDevoxelizeOps.cpp)) | N | Y | Y | Whole file is wrapped in `#if defined(BUILD_CUDA_MODULE) \|\| defined(BUILD_SYCL_MODULE)` — no CPU code path or kernel implementation exists at all. Ported from upstream PVCNN, which was GPU-only by design. |

`BuildSpatialHashTable` has all 3 backends but an asymmetric dtype matrix
(CUDA/SYCL only instantiate `float`; CPU also instantiates `double`) — not a
missing-backend gap, just a dtype gap.

## Usage in Open3D-ML (sibling repo)

- **`trilinear_devoxelize_forward`/`_backward`**: used by `PVCNN`
  (`ml3d/torch/models/pvcnn.py`, `TrilinearDevoxelization` autograd
  `Function`). Import is guarded by
  `if open3d.core.cuda.device_count() > 0: ...`, so Open3D-ML's PVCNN only
  ever calls this on CUDA today — the CPU gap is real in Open3D but currently
  unexercised. Note the guard also means PVCNN doesn't run on SYCL in
  Open3D-ML yet either, even though Open3D itself supports it now — a
  separate, Open3D-ML-side gap.
- **Legacy `knn_search`** (`KnnSearchOps.cpp`'s `open3d::knn_search`): used by
  `PointTransformer` (`ml3d/torch/models/point_transformer.py` and the tf
  equivalent) via `knn_batch`/`queryandgroup` in its attention layers.
  `knn_batch` explicitly does `points = points.cpu(); queries = queries.cpu()`
  before calling — i.e. Open3D-ML already works around the CUDA gap by
  forcing this op onto CPU. This is the one gap with clear evidence of being
  load-bearing (a real perf compromise in a currently-used model), not dead
  code.
- **Legacy `radius_search`/`MultiRadiusSearch`**: no usage found anywhere in
  Open3D-ML (source or GitHub search). Likely superseded by `FixedRadiusSearch`
  (a different, layer-based op with full CPU/CUDA/SYCL support, used by
  KPFCNN via `open3d.ml.torch.layers.FixedRadiusSearch`).
- **`VoxelPooling`**: no usage found anywhere in Open3D-ML. Likely
  superseded by `Voxelize` (full CPU/CUDA/SYCL support; used by
  `SparseConvUnet`/`PointPillars`).

(Separately, `DataProcessing.knn_search` in
`ml3d/datasets/utils/dataprocessing.py`, used by RandLANet, is a different
pure-Python/NumPy static method going through `o3c.nns.NearestNeighborSearch`
— not the PyTorch op, not relevant to this gap.)

## Prioritization takeaway

- Legacy `knn_search` CUDA/SYCL gap: real and actively worked around by
  PointTransformer forcing `.cpu()`. Highest potential value to close, but
  also the most involved fix (needs a CUDA/SYCL nanoflann-equivalent KD-tree,
  not simple plumbing).
- `TrilinearDevoxelize` CPU gap: not currently blocking Open3D-ML (PVCNN
  gates itself to CUDA-only), though closing it would also require an
  Open3D-ML-side change to lift the `cuda.device_count() > 0` guard for SYCL.
- `VoxelPooling` and legacy `radius_search` CUDA/SYCL gaps: no active
  consumer found in Open3D-ML — lowest priority; candidates for leaving as
  documented follow-ups or eventual deprecation rather than new kernel work.

Status: **documented only, no code changes made** for these gaps (per
explicit instruction — deferred, not implemented, this session).
