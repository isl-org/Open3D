# Point cloud smoothing implementation notes

## Goal

Add hand-derived unit-cube reference tests for the legacy smoothers, then
provide matching Tensor point-cloud APIs.

## Decisions and progress

- The legacy reference fixture has eight cube corners in binary-coordinate
  order. Laplacian and Taubin use full fixed k-NN neighborhoods, allowing
  closed-form reference positions.
- MLS uses one displaced cube corner to avoid the eigensystem degeneracy of a
  perfect cube. Bilateral uses explicit outward corner normals.
- Tensor methods preserve arbitrary Tensor point attributes by cloning the
  input and replacing positions (and normals when smoothing produces them).
- Current Tensor implementation delegates its numerical work to the legacy
  reference implementation and returns results to the source device. Device
  kernels remain to be ported for native CUDA/SYCL execution.

## Verification

- `PointCloudSmoothing.*UnitCube*`: passes locally.
- `*SmoothLaplacianUnitCubeReference*`: passes on CPU locally.
